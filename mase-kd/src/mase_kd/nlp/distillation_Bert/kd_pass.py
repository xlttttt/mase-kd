import torch
import torch.nn as nn
from typing import Dict, Any

from .losses import prediction_distillation_loss

class KnowledgeDistillationPass(nn.Module):
    """
    MASE KD Pass (FX Traced Compatible Version: Logits-only Distillation)
    """
    def __init__(
        self, 
        student_model: nn.Module, 
        teacher_model: nn.Module, 
        alpha_kd: float = 1.0, 
        temperature: float = 3.0,
        **kwargs  # Use **kwargs to absorb extra parameters like s_dim, t_layers from the Notebook
    ):
        super().__init__()
        self.student = student_model
        
        # Freeze teacher model parameters to prevent weight updates
        self.teacher = teacher_model
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
        self.alpha_kd = alpha_kd
        self.temperature = temperature

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs) -> Dict[str, Any]:
        # 🌟 Remove redundant parameters automatically passed by newer HF versions
        kwargs.pop("num_items_in_batch", None)

        # 1. Teacher model inference (Native HF Model)
        with torch.no_grad():
            t_outputs = self.teacher(
                input_ids=input_ids, 
                attention_mask=attention_mask
            )

        # 2. Student model inference (FX Graph Traced Model)
        s_outputs = self.student(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels
        )

        # 3. 🌟 Robust Feature Extraction: Compatible with both Object and Dict outputs 🌟
        # Extract Teacher Logits
        t_logits = t_outputs.logits if hasattr(t_outputs, "logits") else t_outputs["logits"]
        
        # Extract Student Logits
        s_logits = s_outputs["logits"] if isinstance(s_outputs, dict) else s_outputs.logits

        # Extract Student Task Loss
        if isinstance(s_outputs, dict):
            s_loss = s_outputs.get("loss", None)
        else:
            s_loss = getattr(s_outputs, "loss", None)

        # 4. Compute Logits-based Distillation Loss
        loss_pred = prediction_distillation_loss(s_logits, t_logits, T=self.temperature)

        # 5. Combine with Original Task Cross-Entropy Loss
        task_loss = s_loss if (labels is not None and s_loss is not None) else torch.tensor(0.0).to(input_ids.device)
        final_loss = task_loss + self.alpha_kd * loss_pred

        return {
            "loss": final_loss,
            "task_loss": task_loss,
            "kd_loss_pred": loss_pred,
            "logits": s_logits
        }