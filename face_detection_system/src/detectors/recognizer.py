"""
Face Recognition — OpenCV SFace Module
"""
import os
import urllib.request
import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from ..utils import get_logger

logger = get_logger("recognizer")

_MODEL_URL = "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

class FaceRecognizer:
    def __init__(self, model_dir: str = "models", download: bool = True):
        self.model_path = os.path.join(model_dir, "face_recognition_sface_2021dec.onnx")
        self._ensure_model(model_dir, download)
        self.recognizer = cv2.FaceRecognizerSF_create(self.model_path, "")
        
        # Database persistence
        self.db_path = os.path.join(model_dir, "enrolled_faces.npy")
        self.database: Dict[str, np.ndarray] = self._load_database()
        
        self.enrolled_names = []
        self.enrolled_matrix = None
        self._build_cache()
        
        # SFace expects 112x112 input
        self.input_size = (112, 112)
        
        # Cosine similarity threshold for SFace (recommended ~0.364)
        self.threshold = 0.364

    def _ensure_model(self, model_dir: str, download: bool):
        if os.path.isfile(self.model_path):
            return
        if not download:
            raise FileNotFoundError(f"SFace model not found at {self.model_path}")
        
        os.makedirs(model_dir, exist_ok=True)
        logger.info(f"Downloading SFace model to {self.model_path} (~36MB)...")
        urllib.request.urlretrieve(_MODEL_URL, self.model_path)
        logger.info("Download complete.")

    def _load_database(self) -> Dict[str, np.ndarray]:
        if os.path.exists(self.db_path):
            try:
                data = np.load(self.db_path, allow_pickle=True).item()
                logger.info(f"Loaded {len(data)} enrolled faces from {self.db_path}")
                return data
            except Exception as e:
                logger.error(f"Failed to load database: {e}")
        return {}

    def _save_database(self):
        try:
            np.save(self.db_path, self.database)
        except Exception as e:
            logger.error(f"Failed to save database: {e}")

    def _build_cache(self):
        """Pre-computes a normalized matrix of all enrolled faces for O(1) vectorized search."""
        if not self.database:
            self.enrolled_names = []
            self.enrolled_matrix = None
            return
            
        self.enrolled_names = list(self.database.keys())
        features = np.array([self.database[n].flatten() for n in self.enrolled_names])
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self.enrolled_matrix = features / norms
        logger.info(f"Rebuilt vector search cache with {len(self.enrolled_names)} identities.")

    def _extract_feature(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        if face_crop is None or face_crop.size == 0:
            return None
        # Resize to expected input size
        resized = cv2.resize(face_crop, self.input_size)
        feature = self.recognizer.feature(resized)
        return feature

    def enroll(self, name: str, face_crop: np.ndarray) -> bool:
        """Enroll a face crop with a given name."""
        feature = self._extract_feature(face_crop)
        if feature is None:
            return False
        self.database[name] = feature
        self._save_database()
        self._build_cache()
        logger.info(f"Enrolled face for '{name}'. Total enrolled: {len(self.database)}")
        return True

    def identify(self, face_crop: np.ndarray) -> Tuple[str, float]:
        """Identify a face crop using vectorized cosine similarity."""
        feature = self._extract_feature(face_crop)
        if feature is None or self.enrolled_matrix is None:
            return "Unknown", 0.0

        feat_flat = feature.flatten()
        norm = np.linalg.norm(feat_flat)
        if norm == 0:
            return "Unknown", 0.0
        feat_norm = feat_flat / norm

        # Vectorized cosine similarity (Dot product of normalized vectors)
        scores = np.dot(self.enrolled_matrix, feat_norm)
        
        best_idx = np.argmax(scores)
        best_score = float(scores[best_idx])
        
        if best_score >= self.threshold:
            return self.enrolled_names[best_idx], best_score
            
        return "Unknown", best_score
