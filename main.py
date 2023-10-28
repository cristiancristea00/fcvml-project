"""
Author: Cristian Cristea

Summary: This module contains the main function.
"""

from pathlib import Path
from time import perf_counter_ns
from timeit import timeit
from typing import Iterator, Final

from colorama import Fore
from prettytable import PrettyTable, MARKDOWN

from face_embedder import EmbedderChoiceType, EmbedderChoice
from face_matcher import FaceMatcher, MatchResult
from utils import IMAGE_EXTENSIONS

CACHE_DATASET: Final[bool] = True
MODEL: Final[EmbedderChoiceType] = EmbedderChoice.FACE

TABLE: Final[PrettyTable] = PrettyTable()

HEADER: Final[str] = '# Fundamentals of Computer Vision and Machine Learning Project\n\n'


def main() -> None:
    """
    The main function.
    """

    reference_dataset = Path('dataset', 'reference')
    test_images = Path('dataset', 'test')

    file_globs: Final[Iterator[Iterator[Path]]] = (test_images.rglob(F'*{ext}') for ext in IMAGE_EXTENSIONS)
    files: Final[list[Path]] = sorted(file for glob in file_globs for file in glob if 'faces' not in str(file))

    TABLE.set_style(MARKDOWN)
    TABLE.field_names = ['Test Image', '1st Match', '1st Score', '2nd Match', '2nd Score', '3rd Match', '3rd Score']

    start: int = perf_counter_ns()
    print_results(files, reference_dataset)
    end: int = perf_counter_ns()

    elapsed: Final[float] = (end - start) / 1e9

    print(F'{Fore.CYAN}Total time: {elapsed:.3f} seconds{Fore.RESET}')

    average: Final[float] = timeit(lambda: get_result(files[0], reference_dataset), number=10) / 10
    print(F'{Fore.CYAN}Average time per image: {average:.3f} seconds{Fore.RESET}')


def print_results(files: list[Path], reference_dataset: Path) -> None:
    """
    Prints the results of the face matching.

    Args:
        files (list[Path]): The paths to the images containing the faces
        reference_dataset (Path): The path to the dataset containing the reference images
    """

    for image in files:
        short_name, relative_path, top = get_result(image, reference_dataset)

        row: list[str] = [F'![{short_name}]({relative_path})']

        print(F'{Fore.GREEN}Test image: {short_name}{Fore.RESET}')
        for match in top:
            print(F'{Fore.WHITE}{match}{Fore.RESET}', end='\n\n')
            row.extend([F'![{match.face_embedding.detection.shorter_path}]({match.face_embedding.detection.relative_path})', match.similarity_str])

        TABLE.add_row(row)

    with open('README.md', 'w', encoding='UTF-8') as readme:
        readme.write(HEADER)
        readme.write(TABLE.get_string())


def get_result(image: Path, reference_dataset: Path) -> tuple[str, str, list[MatchResult]]:
    """
    Gets the result of the face matching.

    Args:
        image (Path): The path to the image containing the face
        reference_dataset (Path): The path to the dataset containing the reference images

    Returns:
        tuple[str, list[MatchResult]]: The name of the image and the top matches
    """
    face_matcher = FaceMatcher(image, reference_dataset, cache=CACHE_DATASET, model=MODEL)

    short_name: Final[str] = F'{image.parent.name}/{image.name}'

    dataset_idx = str(image).find('dataset')
    relative_path = str(image)[dataset_idx:]

    top: Final[list[MatchResult]] = face_matcher.get_top_matches()

    return short_name, relative_path, top


if __name__ == '__main__':
    main()
