"""
Author: Cristian Cristea

Summary: This module contains the main function.
"""

from pathlib import Path
from time import perf_counter_ns
from timeit import timeit
from typing import Iterator, Final

from colorama import Fore

from face_matcher import FaceMatcher, MatchResult
from utils import IMAGE_EXTENSIONS

CACHE_DATASET: Final[bool] = True
LARGE_MODEL: Final[bool] = True


def main() -> None:
    """
    The main function.
    """

    reference_dataset = Path('dataset', 'reference')
    test_images = Path('dataset', 'test')

    file_globs: Final[Iterator[Iterator[Path]]] = (test_images.rglob(F'*{ext}') for ext in IMAGE_EXTENSIONS)
    files: Final[list[Path]] = sorted(file for glob in file_globs for file in glob if 'faces' not in str(file))

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
        short_name, top = get_result(image, reference_dataset)

        print(F'{Fore.GREEN}Test image: {short_name}{Fore.RESET}')
        for match in top:
            print(F'{Fore.WHITE}{match}{Fore.RESET}', end='\n\n')


def get_result(image: Path, reference_dataset: Path) -> tuple[str, list[MatchResult]]:
    """
    Gets the result of the face matching.

    Args:
        image (Path): The path to the image containing the face
        reference_dataset (Path): The path to the dataset containing the reference images

    Returns:
        tuple[str, list[MatchResult]]: The name of the image and the top matches
    """
    face_matcher = FaceMatcher(image, reference_dataset, cache=CACHE_DATASET, large=LARGE_MODEL)

    short_name: Final[str] = F'{image.parent.name}/{image.name}'
    top: Final[list[MatchResult]] = face_matcher.get_top_matches()

    return short_name, top


if __name__ == '__main__':
    main()
