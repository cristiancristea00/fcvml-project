import pickle
from pathlib import Path
from typing import Final, Iterator

from face_embedder import FaceEmbedding, FaceEmbedder


class FaceMatcher:
    _IMAGE_EXTENSIONS: Final[tuple[str, ...]] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

    def __init__(self, image_path: Path, dataset_path: Path) -> None:
        self.face_embedding: Final[FaceEmbedding] = FaceEmbedder.embed_face(image_path)
        self.similarities: Final[list[tuple[FaceEmbedding, float]]] = self._get_similarities(dataset_path)

    def _get_similarities(self, dataset_path: Path) -> list[tuple[FaceEmbedding, float]]:
        path: Final[Path] = Path(dataset_path).resolve()

        if not path.exists():
            raise RuntimeError(F'Dataset {dataset_path} does not exist')

        if not path.is_dir():
            raise RuntimeError(F'Dataset {dataset_path} is not a directory')

        picked_path: Final[Path] = path.with_suffix('.pkl')

        if picked_path.exists():
            with open(picked_path, 'rb') as pickled_file:
                embeddings: list[FaceEmbedding] = pickle.load(pickled_file)
        else:
            file_globs: Final[Iterator[Iterator[Path]]] = (path.rglob(F'*{ext}') for ext in self._IMAGE_EXTENSIONS)
            files: Final[list[Path]] = sorted(file for glob in file_globs for file in glob if 'faces' not in str(file))
            embeddings: list[FaceEmbedding] = [FaceEmbedder.embed_face(file) for file in files]

        similarities: Final[list[tuple[FaceEmbedding, float]]] = list(
            (embedding, FaceEmbedder.compute_similarity(self.face_embedding, embedding))
            for embedding in embeddings
        )

        similarities.sort(key=lambda elem: elem[1], reverse=True)

        with open(picked_path, 'wb') as file:
            pickle.dump(embeddings, file)

        return similarities

    def get_top_matches(self, number: int = 3) -> list[tuple[FaceEmbedding, float]]:
        return self.similarities[:number]
