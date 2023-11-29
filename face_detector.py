from dataclasses import dataclass, astuple
from pathlib import Path
from typing import Final, Iterator

import cv2 as cv
import numpy as np
from mediapipe import Image
from mediapipe.tasks.python import vision
from numpy import ndarray


@dataclass(slots=True, frozen=True)
class FaceBoundingBox:
    start_x: int
    start_y: int
    end_x: int
    end_y: int

    def __iter__(self) -> Iterator[int]:
        return iter(astuple(self))


@dataclass(slots=True, frozen=True)
class FaceDetectionResult:
    image_path: Path
    bounding_box: FaceBoundingBox
    face_path: Path
    confidence: float


class FaceDetector:
    _DETECTOR = vision.FaceDetector.create_from_model_path('BlazeFace.tflite')

    @classmethod
    def detect_face(cls, image_path: Path) -> FaceDetectionResult:
        path = Path(image_path)

        if not path.exists():
            raise RuntimeError(F'Image {image_path} does not exist')

        if not path.is_file():
            raise RuntimeError(F'Image {image_path} is not a file')

        path = path.resolve()

        image: Final[Image] = Image.create_from_file(str(path))
        detection_results = cls._DETECTOR.detect(image)

        if not detection_results.detections:
            raise RuntimeError(F'No face detected in {image_path}')

        detection = detection_results.detections[0]
        bounding_box = detection.bounding_box

        start_x: Final[int] = bounding_box.origin_x
        start_y: Final[int] = bounding_box.origin_y
        end_x: Final[int] = bounding_box.origin_x + bounding_box.width
        end_y: Final[int] = bounding_box.origin_y + bounding_box.height

        face_bounding_box: Final[FaceBoundingBox] = FaceBoundingBox(
            start_x=start_x,
            start_y=start_y,
            end_x=end_x,
            end_y=end_y,
        )

        confidence: Final[float] = detection.categories[0].score

        original: Final[ndarray] = np.copy(image.numpy_view())
        face: Final[ndarray] = original[start_y:end_y, start_x:end_x]

        folder: Final[Path] = Path(image_path.parent, 'faces')
        folder.mkdir(exist_ok=True)

        face_path: Final[Path] = folder / image_path.name
        conv_face: Final[ndarray] = cv.cvtColor(face, cv.COLOR_RGB2BGR)
        cv.imwrite(str(face_path), conv_face)

        result: Final[FaceDetectionResult] = FaceDetectionResult(
            image_path=path,
            bounding_box=face_bounding_box,
            face_path=face_path,
            confidence=confidence,
        )

        return result
