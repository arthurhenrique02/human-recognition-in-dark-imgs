from ultralytics import YOLO
import cv2
import numpy as np


class ImageManipulation:
    """
    Class to manipulate images using YOLO model
    The main goal is to track objects in images
    """

    __model = YOLO("yolov8n.pt")

    def track_image(self, path: str):
        """
        Track objects in image

        :param path: str
            Path to image
        :return: np.ndarray
            Image with objects tracked
        """
        # get image and track it
        # img_cap = cv2.imread(path)
        results = self.__model.track(path, persist=True)
        plotted_frame = results[0].plot()

        return plotted_frame

    def show_image_with_objects_tracked(self, img: np.ndarray) -> None:
        """
        Show image with objects tracked

        :param img: np.ndarray
            Image to be shown
        """
        cv2.imshow("frame", img)
        if cv2.waitKey(5000) and 0xFF == ord("q"):
            cv2.destroyAllWindows()

    def enchance_image_light(self, img_input: str) -> np.ndarray:
        """
        Enchance image light

        :param img: np.ndarray
            Image to be enchanced
        :param value: int
            Value to be added to image
        :return: np.ndarray
            Enchanced image
        """
        img = cv2.imread(img_input, 0)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(15, 15))

        return clahe.apply(img)

    def enchance_image_contrast(
        self, img: np.ndarray, value: float = 1.5
    ) -> np.ndarray:
        """
        Enchance image contrast

        :param img: np.ndarray
            Image to be enchanced
        :param value: float
            Value to be multiplied to image
        :return: np.ndarray
            Enchanced image
        """
        img = cv2.multiply(img, np.array([value]))
        return img
