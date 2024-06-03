import cv2

from utils.ImgManipulation import ImageManipulation
from utils.enchance import dehaze


img_manipulation = ImageManipulation()

img = cv2.imread("images/people/2015_06829.png")

img_d = dehaze(img, w=5, alpha=0.1, omega=0.73, p=0.1, eps=1e-3, reduce=False)

created_enchanced_img = cv2.imwrite("2015_06829_enchanced.png", img_d)

if created_enchanced_img:
    tracked_imgs_objs = img_manipulation.track_image("2015_06829_enchanced.png")

    img_manipulation.show_image_with_objects_tracked(tracked_imgs_objs)
