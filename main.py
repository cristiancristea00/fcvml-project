from pathlib import Path
from typing import Iterator, Final

from face_matcher import FaceMatcher, MatchResult
from utils import IMAGE_EXTENSIONS

from colorama import Fore


def get_result(image: Path, reference_dataset: Path) -> tuple[str, list[MatchResult]]:
    face_matcher = FaceMatcher(image, reference_dataset)

    short_name: Final[str] = F'{image.parent.name}/{image.name}'
    top: Final[list[MatchResult]] = face_matcher.get_top_matches()

    return short_name, top


def main() -> None:
    reference_dataset = Path('dataset', 'reference')
    test_images = Path('dataset', 'test')

    file_globs: Final[Iterator[Iterator[Path]]] = (test_images.rglob(F'*{ext}') for ext in IMAGE_EXTENSIONS)
    files: Final[list[Path]] = sorted(file for glob in file_globs for file in glob if 'faces' not in str(file))

    for image in files:
        short_name, top = get_result(image, reference_dataset)

        print(F'{Fore.GREEN}Test image: {short_name}{Fore.RESET}')
        for match in top:
            print(F'{Fore.WHITE}{match}{Fore.RESET}', end='\n\n')


if __name__ == '__main__':
    main()
