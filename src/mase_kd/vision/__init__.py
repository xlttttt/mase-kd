"""Vision KD pipelines."""

from .yolo_kd import YOLOLogitsDistiller, YOLOLogitsKDOutput

__all__ = ["YOLOLogitsDistiller", "YOLOLogitsKDOutput"]
