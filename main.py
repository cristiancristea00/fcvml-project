from pathlib import Path

from face_matcher import FaceMatcher


def main():
    test_image = Path('dataset', 'test', 'subject1', '1.jpg')
    test_dataset = Path('dataset', 'reference')
    face_matcher = FaceMatcher(test_image, test_dataset)
    top = face_matcher.get_top_matches()
    print(top)


if __name__ == '__main__':
    main()
