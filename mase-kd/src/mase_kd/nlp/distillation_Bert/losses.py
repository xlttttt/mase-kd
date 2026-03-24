import torch
import torch.nn as nn
import torch.nn.functional as F

def attention_distillation_loss(student_atts, teacher_atts, layer_mapping):
    """
    1. Attention Matrix Distillation (MSE Loss)
    Fits the unnormalized attention matrices to help the student learn the teacher's attention distribution.
    """
    loss = 0.0
    for s_idx, t_idx in layer_mapping.items():
        s_att = student_atts[s_idx]
        t_att = teacher_atts[t_idx]
        
        # Calculate Mean Squared Error (MSE) loss for attention matrices
        loss += F.mse_loss(s_att, t_att)
        
    return loss

class HiddenDistillationLoss(nn.Module):
    """
    2. Hidden State Distillation (MSE Loss with Linear Projection)
    Since the student's hidden dimension is usually smaller than the teacher's (e.g., 312 vs 768),
    a learnable linear projection layer (W_h) is required to align the dimensions before computing MSE.
    """
    def __init__(self, student_dim, teacher_dim):
        super().__init__()
        # Core: Dimension alignment matrix Wh (updated automatically during backpropagation)
        self.fit_dense = nn.Linear(student_dim, teacher_dim)

    def forward(self, student_hiddens, teacher_hiddens, layer_mapping):
        loss = 0.0
        for s_idx, t_idx in layer_mapping.items():
            s_hid = student_hiddens[s_idx]
            t_hid = teacher_hiddens[t_idx]
            
            # First, project the student's hidden state to match the teacher's dimension
            projected_s_hid = self.fit_dense(s_hid)
            
            # Then, calculate the MSE loss
            loss += F.mse_loss(projected_s_hid, t_hid)
            
        return loss

def prediction_distillation_loss(student_logits, teacher_logits, T=3.0):
    """
    3. Logits Distillation (KL Divergence with Temperature T)
    Soft label distillation. A higher Temperature (T) makes the teacher's probability 
    distribution smoother, allowing the student to learn more 'dark knowledge'.
    """
    # Teacher generates smooth targets using Softmax with Temperature
    soft_targets = F.softmax(teacher_logits / T, dim=-1)
    
    # Student generates log probabilities using LogSoftmax with Temperature
    soft_log_probs = F.log_softmax(student_logits / T, dim=-1)
    
    # Calculate KL Divergence. 
    # Note: Must multiply by T^2 to keep the gradient scale consistent with standard cross-entropy!
    loss = F.kl_div(soft_log_probs, soft_targets, reduction='batchmean') * (T**2)
    
    return loss