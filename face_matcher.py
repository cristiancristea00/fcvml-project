import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterator

from face_embedder import FaceEmbedding, FaceEmbedder
from utils import IMAGE_EXTENSIONS


@dataclass(slots=True, frozen=True)
class MatchResult:
    face_embedding: FaceEmbedding
    similarity: float

    def __repr__(self) -> str:
        return F'{self.face_embedding}\nSimilarity: {self.similarity * 100:.2f}%'


class FaceMatcher:
    def __init__(self, image_path: Path, dataset_path: Path, cache: bool = True) -> None:
        self.face_embedding: Final[FaceEmbedding] = FaceEmbedder.embed_face(image_path)
        self.similarities: Final[list[MatchResult]] = self._get_similarities(dataset_path, cache)

    def _get_similarities(self, dataset_path: Path, cache: bool) -> list[MatchResult]:
        path: Final[Path] = Path(dataset_path).resolve()

        if not path.exists():
            raise RuntimeError(F'Dataset {dataset_path} does not exist')

        if not path.is_dir():
            raise RuntimeError(F'Dataset {dataset_path} is not a directory')

        picked_path: Final[Path] = path.with_suffix('.pkl')

        if picked_path.exists() and cache:
            with picked_path.open('rb') as pickled_file:
                embeddings: list[FaceEmbedding] = pickle.load(pickled_file)
        else:
            file_globs: Final[Iterator[Iterator[Path]]] = (path.rglob(F'*{ext}') for ext in IMAGE_EXTENSIONS)
            files: Final[Iterator[Path]] = (file for glob in file_globs for file in glob if 'faces' not in str(file))
            embeddings: list[FaceEmbedding] = [FaceEmbedder.embed_face(file) for file in files]

            with picked_path.open('wb') as pickled_file:
                pickle.dump(embeddings, pickled_file)

        similarities: Final[list[tuple[FaceEmbedding, float]]] = list(
            (embedding, FaceEmbedder.compute_similarity(self.face_embedding, embedding))
            for embedding in embeddings
        )

        result: Final[list[MatchResult]] = [MatchResult(face_embedding, similarity) for face_embedding, similarity in similarities]

        result.sort(key=lambda elem: elem.similarity, reverse=True)

        return result

    def get_top_matches(self, number: int = 3) -> list[MatchResult]:
        return self.similarities[:number]
