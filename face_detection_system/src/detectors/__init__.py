from .base import BaseDetector, Detection
from .haar_detector import HaarCascadeDetector
from .dnn_detector import DNNDetector

__all__ = ["BaseDetector", "Detection", "HaarCascadeDetector", "DNNDetector"]
