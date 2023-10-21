from dataclasses import dataclass
from pathlib import Path
from typing import Final

from mediapipe import Image
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers import Embedding
from numpy import ndarray

from face_detector import FaceDetectionResult, FaceDetector


@dataclass(slots=True, frozen=True)
class FaceEmbedding:
    detection: FaceDetectionResult
    embedding: Embedding

    def __repr__(self) -> str:
        embedding_vals: Final[ndarray] = self.embedding.embedding
        embedding: Final[str] = F'({embedding_vals[0]:.3f}, {embedding_vals[1]:.3f}, {embedding_vals[2]:.3f}, ..., {embedding_vals[-2]:.3f}, {embedding_vals[-1]:.3f})'

        return F'Face detection: {self.detection}\nEmbedding: {embedding}'


class FaceEmbedder:
    _EMBEDDER = vision.ImageEmbedder.create_from_model_path('MobileNet-V3-Large.tflite')

    @classmethod
    def embed_face(cls, image_path: Path) -> FaceEmbedding:
        detected_face: Final[FaceDetectionResult] = FaceDetector.detect_face(image_path)
        face: Final[Image] = Image.create_from_file(str(detected_face.face_path))
        embedding: Final[Embedding] = cls._EMBEDDER.embed(face).embeddings[0]

        result: Final[FaceEmbedding] = FaceEmbedding(
            detection=detected_face,
            embedding=embedding,
        )

        return result

    @classmethod
    def compute_similarity(cls, face1: FaceEmbedding, face2: FaceEmbedding) -> float:
        return vision.ImageEmbedder.cosine_similarity(face1.embedding, face2.embedding)
