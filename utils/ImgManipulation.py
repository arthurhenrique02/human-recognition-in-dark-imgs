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
        results = self.__model.track(path, persist=True, classes=0)
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
