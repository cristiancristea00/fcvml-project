"""
Author: Cristian Cristea

Summary: This module contains the FaceMatcher class, which is used to compare
         faces.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterator

from face_embedder import FaceEmbedding, FaceEmbedder, EmbedderChoiceType, EmbedderChoice
from utils import IMAGE_EXTENSIONS


@dataclass(slots=True, frozen=True)
class MatchResult:
    """
    Represents the result of a face match.
    """

    face_embedding: FaceEmbedding
    similarity: float

    @property
    def similarity_str(self) -> str:
        """
        Returns a string representation of the similarity.

        Returns:
            str: A string representation of the similarity
        """
        return F'{self.similarity:.3f}'

    def __repr__(self) -> str:
        """
        Returns a string representation of the face match result.

        Returns:
            str: A string representation of the face match result
        """
        return F'{self.face_embedding}\nSimilarity: {self.similarity_str}'


class FaceMatcher:
    """
    Class used to compare faces.
    """

    def __init__(self, image_path: Path, dataset_path: Path, cache: bool = True, model: EmbedderChoiceType = EmbedderChoice.FACE) -> None:
        """
        Creates a FaceMatcher object.

        Args:
            image_path (Path): The path to the image containing the face
            dataset_path (Path): The path to the dataset containing the reference images
            cache (bool, optional): Whether to cache the embeddings of the reference images. Defaults to True.
            model (EmbedderChoiceType, optional): The model to use for the embedding. Defaults to EmbedderChoice.FACE.
        """

        self.dataset_path: Final[Path] = dataset_path
        self.model: Final[EmbedderChoiceType] = model
        self.face_embedding: Final[FaceEmbedding] = FaceEmbedder.embed_face(image_path, self.model)
        self.similarities: Final[list[MatchResult]] = self._get_similarities(cache)

    def _get_similarities(self, cache: bool) -> list[MatchResult]:
        """
        Gets the similarities between the face in the image and the faces in the
        dataset.

        Args:
            cache (bool): Whether to cache the embeddings of the reference images

        Raises:
            RuntimeError: If the dataset does not exist
            RuntimeError: If the dataset is not a directory

        Returns:
            list[MatchResult]: The similarities between the face in the image and the faces in the dataset
        """

        path: Final[Path] = Path(self.dataset_path).resolve()

        if not path.exists():
            raise RuntimeError(F'Dataset {self.dataset_path} does not exist')

        if not path.is_dir():
            raise RuntimeError(F'Dataset {self.dataset_path} is not a directory')

        pickled_path: Final[Path] = path.with_suffix('.pkl')

        if pickled_path.exists() and cache:
            with pickled_path.open('rb') as pickled_file:
                embeddings: list[FaceEmbedding] = pickle.load(pickled_file)
        else:
            embeddings: list[FaceEmbedding] = self.get_embeddings(path)
            self.write_embeddings(pickled_path, embeddings)

        try:
            similarities: list[tuple[FaceEmbedding, float]] = self.get_similarities(embeddings)
        except ValueError:
            embeddings: list[FaceEmbedding] = self.get_embeddings(path)
            self.write_embeddings(pickled_path, embeddings)
            similarities: list[tuple[FaceEmbedding, float]] = self.get_similarities(embeddings)

        result: Final[list[MatchResult]] = [MatchResult(face_embedding, similarity) for face_embedding, similarity in similarities]

        result.sort(key=lambda elem: elem.similarity, reverse=True)

        return result

    def get_top_matches(self, number: int = 3) -> list[MatchResult]:
        """
        Gets the top N matches.

        Args:
            number (int, optional): The number of matches to return. Defaults to 3.

        Returns:
            list[MatchResult]: The top N matches
        """

        return self.similarities[:number]

    def get_embeddings(self, path: Path) -> list[FaceEmbedding]:
        """
        Gets the embeddings of the faces in the dataset.

        Args:
            path (Path): The path to the dataset

        Returns:
            list[FaceEmbedding]: The embeddings of the faces in the dataset
        """

        file_globs: Final[Iterator[Iterator[Path]]] = (path.rglob(F'*{ext}') for ext in IMAGE_EXTENSIONS)
        files: Final[Iterator[Path]] = (file for glob in file_globs for file in glob if 'faces' not in str(file))
        return [FaceEmbedder.embed_face(file, self.model) for file in files]

    def get_similarities(self, embeddings: list[FaceEmbedding]) -> list[tuple[FaceEmbedding, float]]:
        """
        Gets the similarities between the face in the image and the faces in the
        dataset.

        Args:
            embeddings (list[FaceEmbedding]): The embeddings of the faces in the dataset

        Returns:
            list[tuple[FaceEmbedding, float]]: The similarities between the face in the image and the faces in the dataset
        """

        return list(
            (embedding, FaceEmbedder.compute_similarity(self.face_embedding, embedding))
            for embedding in embeddings
        )

    @classmethod
    def write_embeddings(cls, pickled_path: Path, embeddings: list[FaceEmbedding]) -> None:
        """
        Writes the embeddings to a file.

        Args:
            pickled_path (Path): The path to the file
            embeddings (list[FaceEmbedding]): The embeddings to write
        """

        with pickled_path.open('wb') as pickled_file:
            pickle.dump(embeddings, pickled_file)
