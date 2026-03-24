import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple

from .losses import prediction_distillation_loss, HiddenDistillationLoss
from .mapping import generate_layer_mapping

# ==========================================
# 1. Underlying Core Engine (Internal use only, called by the Pass)
# ==========================================
class _KnowledgeDistillationWrapper(nn.Module):
    """
    Internal KD Wrapper: Handles static graph hooks, teacher inference, 
    and joint loss computation (Logits + Hidden States + Attention).
    """
    def __init__(
        self, 
        student_model: nn.Module, 
        teacher_model: nn.Module, 
        s_dim: int, 
        t_dim: int, 
        s_layers: int, 
        t_layers: int,
        alpha_kd: float = 1.0, 
        alpha_hidden: float = 1.0,
        alpha_attn: float = 1.0,   # New parameter for Attention KD weight
        temperature: float = 3.0,
    ):
        super().__init__()
        self.student = student_model
        
        # Freeze teacher model parameters to prevent them from being updated by the optimizer
        self.teacher = teacher_model
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
        # Loss weights and hyperparameters
        self.alpha_kd = alpha_kd
        self.alpha_hidden = alpha_hidden
        self.alpha_attn = alpha_attn
        self.temperature = temperature
        
        # Module initialization
        self.hidden_loss_fn = HiddenDistillationLoss(s_dim, t_dim)
        self.layer_mapping = generate_layer_mapping(s_layers, t_layers)
        
        # Containers to hold intermediate features captured by hooks
        self.student_hidden_states = []
        self.student_attentions = []
        
        self._register_student_hooks()

    def _register_student_hooks(self):
        """
        Register hooks to bypass static graph limitations and extract 
        both hidden states and attention matrices during the forward pass.
        """
        # Hook for Hidden States (applied after LayerNorm)
        def hidden_hook_fn(module, input, output):
            hidden_state = output[0] if isinstance(output, tuple) else output
            self.student_hidden_states.append(hidden_state)

        # Hook for Attention Matrices (applied BEFORE Dropout)
        def attn_hook_fn(module, args):
            # args[0] is the input to the Dropout layer, which in BERT is exactly 
            # the attention probability matrix right after Softmax.
            attn_matrix = args[0]
            self.student_attentions.append(attn_matrix)

        hooked_hidden = 0
        hooked_attn = 0
        
        for name, module in self.student.named_modules():
            # 1. Capture Hidden States (Output LayerNorms)
            if "LayerNorm" in module.__class__.__name__ and "encoder" in name and "output" in name:
                module.register_forward_hook(hidden_hook_fn)
                hooked_hidden += 1
                
            # 2. Capture Attention Matrices (Self-Attention Dropout)
            # Using a pre_hook ensures we get the clean matrix BEFORE dropout masks are applied
            elif "Dropout" in module.__class__.__name__ and "attention.self" in name:
                module.register_forward_pre_hook(attn_hook_fn)
                hooked_attn += 1
                
        print(f"[KD Pass] Hooks successfully registered: {hooked_hidden} Hidden layers, {hooked_attn} Attention layers.")

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs) -> Dict[str, Any]:
        kwargs.pop("num_items_in_batch", None)
        
        # Clear residual features from the previous forward pass
        self.student_hidden_states.clear() 
        self.student_attentions.clear()

        # --------------------------------------------------
        # Teacher Inference
        # --------------------------------------------------
        with torch.no_grad():
            t_outputs = self.teacher(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True      # <--- Request attention matrices from the teacher
            )

        # --------------------------------------------------
        # Student Inference (Triggers the hooks)
        # --------------------------------------------------
        s_outputs = self.student(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels
        )

        # Extract basic features (Logits and Task Loss)
        t_logits = t_outputs.logits if hasattr(t_outputs, "logits") else t_outputs["logits"]
        s_logits = s_outputs["logits"] if isinstance(s_outputs, dict) else s_outputs.logits
        s_loss = s_outputs.get("loss", None) if isinstance(s_outputs, dict) else getattr(s_outputs, "loss", None)

        # --------------------------------------------------
        # Compute Distillation Losses
        # --------------------------------------------------
        
        # 1. Logits KD Loss
        loss_pred = prediction_distillation_loss(s_logits, t_logits, T=self.temperature)
        
        # 2. Hidden States KD Loss
        t_hiddens = t_outputs.hidden_states[1:] # Discard embedding layer output
        s_hiddens = self.student_hidden_states  
        
        if len(s_hiddens) > 0:
            loss_hid = self.hidden_loss_fn(s_hiddens, t_hiddens, self.layer_mapping)
        else:
            loss_hid = torch.tensor(0.0).to(input_ids.device)

        # 3. Attention KD Loss (Mean Squared Error)
        t_attentions = t_outputs.attentions
        s_attentions = self.student_attentions
        loss_attn = torch.tensor(0.0).to(input_ids.device)
        
        if len(s_attentions) > 0 and t_attentions is not None:
            # Map the student's attention layers to the corresponding teacher's attention layers
            for s_idx, t_idx in enumerate(self.layer_mapping):
                if s_idx < len(s_attentions) and t_idx < len(t_attentions):
                    loss_attn += F.mse_loss(s_attentions[s_idx], t_attentions[t_idx])

        # --------------------------------------------------
        # Final Joint Loss Computation
        # --------------------------------------------------
        task_loss = s_loss if (labels is not None and s_loss is not None) else torch.tensor(0.0).to(input_ids.device)
        
        final_loss = task_loss + (self.alpha_kd * loss_pred) + (self.alpha_hidden * loss_hid) + (self.alpha_attn * loss_attn)

        return {
            "loss": final_loss,
            "task_loss": task_loss,
            "kd_loss": loss_pred + loss_hid + loss_attn,
            "logits": s_logits
        }

# ==========================================
# 2. Standardized MASE Pass Interface Exposed to the User
# ==========================================
def kd_transform_pass(
    student_graph_or_model: Any, 
    teacher_model: nn.Module, 
    config: Dict[str, Any]
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Standard Knowledge Distillation Pass for MASE.
    
    Args:
        student_graph_or_model: Pruned or quantized student model (MaseGraph or nn.Module)
        teacher_model: Full-precision teacher model
        config: Dictionary containing hyperparameter configurations for distillation
        
    Returns:
        A tuple containing the wrapped KD model and a metadata dictionary of execution info
    """
    print("[KD Pass] Initializing Knowledge Distillation graph transformation...")
    
    # Compatible with MASE's MaseGraph object
    student_nn_model = student_graph_or_model.model if hasattr(student_graph_or_model, "model") else student_graph_or_model
    
    # Parse configuration parameters (with default values)
    s_dim = config.get("s_dim", 128)
    t_dim = config.get("t_dim", 768)
    s_layers = config.get("s_layers", 2)
    t_layers = config.get("t_layers", 12)
    alpha_kd = config.get("alpha_kd", 1.0)
    alpha_hidden = config.get("alpha_hidden", 1.0)
    alpha_attn = config.get("alpha_attn", 1.0)     # Parse new attention weight config
    temperature = config.get("temperature", 3.0)

    # Instantiate the core wrapper
    kd_wrapped_model = _KnowledgeDistillationWrapper(
        student_model=student_nn_model,
        teacher_model=teacher_model,
        s_dim=s_dim,
        t_dim=t_dim,
        s_layers=s_layers,
        t_layers=t_layers,
        alpha_kd=alpha_kd,
        alpha_hidden=alpha_hidden,
        alpha_attn=alpha_attn,
        temperature=temperature
    )
    
    # Simulate MASE Pass return format: (modified model, state/metadata info)
    info = {
        "pass_name": "kd_transform_pass",
        "teacher_layers": t_layers,
        "student_layers": s_layers,
        "hooks_registered": True,
        "attention_distillation": True
    }
    
    return kd_wrapped_model, info