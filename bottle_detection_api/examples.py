import cv2

from bottle_detection import find_bottle, draw_on_image, draw_on_video, draw_on_camera

image = cv2.imread("test_image.jpg")
bottle = find_bottle(image)
print("found:", bottle["found"])
print("bottle center:", bottle["bottle_center"])
print("cap center:", bottle["cap_center"])
print("sure:", bottle["sure"])

draw_on_image("test_image.jpg", "result_image.jpg")
print("saved result_image.jpg")

print("drawing on test_video.mp4, this can take a few minutes")
draw_on_video("test_video.mp4", "result_video.mp4")
print("saved result_video.mp4")

print("opening the camera, press any key to stop")
draw_on_camera(0)
