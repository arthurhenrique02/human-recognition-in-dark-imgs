# from ultralytics import YOLO
# import cv2

# model = YOLO("yolov8n.pt")

# PATH = "images/teste4.jpeg"

# img_cap = cv2.imread(PATH)

# results = model.track(img_cap, persist=True)

# frame = results[0].plot()

# cv2.imshow("frame", frame)

# # input("Press Enter to continue...")
# if cv2.waitKey(25000) and 0xFF == ord("q"):
#     cv2.destroyAllWindows()

from utils.ImgManipulation import ImageManipulation
import cv2

img_manipulation = ImageManipulation()

img = img_manipulation.enchance_image_light("images/people/2015_06851.jpg")

a = cv2.imwrite("2015_06851_enchanced.jpg", img)

print(a)

if a:
    b = img_manipulation.track_image("2015_06851_enchanced.jpg")

    img_manipulation.show_image_with_objects_tracked(b)

# img_manipulation.show_image_with_objects_tracked(img)

# img = img_manipulation.track_image("images/people/2015_06851.jpg")

# tracked_img = img_manipulation.track_image(img)

# img_manipulation.show_image_with_objects_tracked(img)
