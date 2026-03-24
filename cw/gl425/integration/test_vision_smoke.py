import torch
from torch import nn

from mase_kd.core.losses import DistillationLossConfig
from mase_kd.vision.yolo_kd import YOLOLogitsDistiller


class TinyVisionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, 32),
            nn.ReLU(),
            nn.Linear(32, 6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def test_yolo_logits_kd_single_train_step() -> None:
    teacher = TinyVisionModel()
    student = TinyVisionModel()

    distiller = YOLOLogitsDistiller(
        teacher=teacher,
        student=student,
        kd_config=DistillationLossConfig(alpha=0.7, temperature=2.0),
    )

    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    batch = {
        "images": torch.randn(4, 3, 8, 8),
        "targets": torch.randint(0, 6, (4,)),
    }

    losses = distiller.train_step(batch=batch, optimizer=optimizer)

    assert losses.total_loss > 0
    assert losses.soft_loss >= 0
