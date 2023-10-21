"""
Author: Cristian Cristea

Summary: This module contains the FaceEmbedder class, which is used to create
         embeddings for faces.
"""

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
    """
    Represents the result of a face embedding.
    """

    detection: FaceDetectionResult
    embedding: Embedding

    def __repr__(self) -> str:
        """
        Returns a string representation of the face embedding.

        Returns:
            str: A string representation of the face embedding
        """

        embedding_vals: Final[ndarray] = self.embedding.embedding
        embedding: Final[str] = F'({embedding_vals[0]:.3f}, {embedding_vals[1]:.3f}, {embedding_vals[2]:.3f}, ..., {embedding_vals[-2]:.3f}, {embedding_vals[-1]:.3f})'

        return F'Face detection: {self.detection}\nEmbedding: {embedding}'


class FaceEmbedder:
    """
    Class used to create embeddings for faces.
    """

    _EMBEDDER = vision.ImageEmbedder.create_from_model_path('MobileNet-V3-Large.tflite')

    @classmethod
    def embed_face(cls, image_path: Path) -> FaceEmbedding:
        """
        Creates an embedding for a face.

        Args:
            image_path (Path): The path to the image containing the face

        Returns:
            FaceEmbedding: The face embedding
        """

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
        """
        Computes the similarity between two faces.

        Args:
            face1 (FaceEmbedding): The first face
            face2 (FaceEmbedding): The second face

        Returns:
            float: The similarity between the two faces
        """

        return vision.ImageEmbedder.cosine_similarity(face1.embedding, face2.embedding)
