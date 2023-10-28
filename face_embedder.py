"""
Author: Cristian Cristea

Summary: This module contains the FaceEmbedder class, which is used to create
         embeddings for faces.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Final, TypeAlias, Literal, assert_never

from mediapipe import Image
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers import Embedding
from numpy import ndarray

from face_detector import FaceDetectionResult, FaceDetector


class EmbedderChoice(Enum):
    """
    Enum used to represent the available face embedding models.
    """

    LARGE = 'MobileNet-V3-Large.tflite'
    SMALL = 'MobileNet-V3-Small.tflite'
    FACE = 'FaceNet.tflite'


EmbedderChoiceType: TypeAlias = Literal[EmbedderChoice.LARGE, EmbedderChoice.SMALL, EmbedderChoice.FACE]


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
        embedding: Final[
            str] = F'({embedding_vals[0]:.3f}, {embedding_vals[1]:.3f}, {embedding_vals[2]:.3f}, ..., {embedding_vals[-2]:.3f}, {embedding_vals[-1]:.3f})'

        return F'Face detection: {self.detection}\nEmbedding: {embedding}'


class FaceEmbedder:
    """
    Class used to create embeddings for faces.
    """

    _MODELS_DIR: Final[Path] = Path('models')

    _EMBEDDER_LARGE = vision.ImageEmbedder.create_from_model_path(str(_MODELS_DIR / EmbedderChoice.LARGE.value))
    _EMBEDDER_SMALL = vision.ImageEmbedder.create_from_model_path(str(_MODELS_DIR / EmbedderChoice.SMALL.value))
    _EMBEDDER_FACE = vision.ImageEmbedder.create_from_model_path(str(_MODELS_DIR / EmbedderChoice.FACE.value))

    @classmethod
    def embed_face(cls, image_path: Path, model: EmbedderChoiceType = EmbedderChoice.FACE) -> FaceEmbedding:
        """
        Creates an embedding for a face.

        Args:
            image_path (Path): The path to the image containing the face
            model (EmbedderChoiceType, optional): The model to use for the embedding. Defaults to EmbedderChoice.FACE.

        Returns:
            FaceEmbedding: The face embedding
        """

        detected_face: Final[FaceDetectionResult] = FaceDetector.detect_face(image_path)
        face: Final[Image] = Image.create_from_file(str(detected_face.face_path))

        match model:
            case EmbedderChoice.LARGE:
                embedding: Final[Embedding] = cls._EMBEDDER_LARGE.embed(face).embeddings[0]
            case EmbedderChoice.SMALL:
                embedding: Final[Embedding] = cls._EMBEDDER_SMALL.embed(face).embeddings[0]
            case EmbedderChoice.FACE:
                embedding: Final[Embedding] = cls._EMBEDDER_FACE.embed(face).embeddings[0]
            case _:
                assert_never(model)

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
