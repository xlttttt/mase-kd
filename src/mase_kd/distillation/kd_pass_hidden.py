import torch
import torch.nn as nn
from typing import Dict, Any, Tuple

from .losses import prediction_distillation_loss, HiddenDistillationLoss
from .mapping import generate_layer_mapping

# ==========================================
# 1. Underlying Core Engine (Internal use only, called by the Pass)
# ==========================================
class _KnowledgeDistillationWrapper(nn.Module):
    """
    Internal KD Wrapper: Handles static graph hooks, teacher inference, and joint loss computation.
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
        temperature: float = 3.0,
    ):
        super().__init__()
        self.student = student_model
        
        # Freeze teacher model parameters to prevent them from being updated by the optimizer
        self.teacher = teacher_model
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
        self.alpha_kd = alpha_kd
        self.alpha_hidden = alpha_hidden
        self.temperature = temperature
        
        # Module initialization
        self.hidden_loss_fn = HiddenDistillationLoss(s_dim, t_dim)
        self.layer_mapping = generate_layer_mapping(s_layers, t_layers)
        
        self.student_hidden_states = []
        self._register_student_hooks()

    def _register_student_hooks(self):
        """Register hooks to bypass static graph limitations and extract hidden states"""
        def hook_fn(module, input, output):
            hidden_state = output[0] if isinstance(output, tuple) else output
            self.student_hidden_states.append(hidden_state)

        hooked_layers = 0
        for name, module in self.student.named_modules():
            if "LayerNorm" in module.__class__.__name__ and "encoder" in name and "output" in name:
                module.register_forward_hook(hook_fn)
                hooked_layers += 1
        print(f"[KD Pass] Successfully registered {hooked_layers} feature hooks on the student model.")

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs) -> Dict[str, Any]:
        kwargs.pop("num_items_in_batch", None)
        self.student_hidden_states.clear() # Clear residual features from the previous forward pass

        # Teacher inference
        with torch.no_grad():
            t_outputs = self.teacher(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                output_hidden_states=True  
            )

        # Student inference (triggers hooks)
        s_outputs = self.student(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels
        )

        # Extract features
        t_logits = t_outputs.logits if hasattr(t_outputs, "logits") else t_outputs["logits"]
        s_logits = s_outputs["logits"] if isinstance(s_outputs, dict) else s_outputs.logits
        s_loss = s_outputs.get("loss", None) if isinstance(s_outputs, dict) else getattr(s_outputs, "loss", None)

        # Compute various losses
        loss_pred = prediction_distillation_loss(s_logits, t_logits, T=self.temperature)
        
        t_hiddens = t_outputs.hidden_states[1:] 
        s_hiddens = self.student_hidden_states  
        
        loss_hid = self.hidden_loss_fn(s_hiddens, t_hiddens, self.layer_mapping) if len(s_hiddens) > 0 else torch.tensor(0.0).to(input_ids.device)

        task_loss = s_loss if (labels is not None and s_loss is not None) else torch.tensor(0.0).to(input_ids.device)
        final_loss = task_loss + self.alpha_kd * loss_pred + self.alpha_hidden * loss_hid

        return {
            "loss": final_loss,
            "task_loss": task_loss,
            "kd_loss": loss_pred + loss_hid,
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
        A tuple containing the wrapped KD model and a metadata dictionary of execution info (compliant with MASE Pass specifications)
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
        temperature=temperature
    )
    
    # Simulate MASE Pass return format: (modified model, state/metadata info)
    info = {
        "pass_name": "kd_transform_pass",
        "teacher_layers": t_layers,
        "student_layers": s_layers,
        "hooks_registered": True
    }
    
    return kd_wrapped_model, info