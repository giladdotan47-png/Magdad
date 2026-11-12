# imports
import math
import os, sys
import cv2
from ultralytics import YOLO

# system adjustments
this_file = os.path.abspath(__file__)
folder = os.path.dirname(this_file)
running_on_mac = sys.platform == 'darwin'

# load model
if running_on_mac:
    model = YOLO(folder + "/model_mac.mlpackage", task="segment")
else:
    model = YOLO(folder + '/model_windows.pt')


# main function, used function are below
def find_bottle(image):
    mask = find_mask(image)
    if mask is None:
        return {"found": False, "bottle_center": None, "cap_center": None, "sure": False}

    bottle_points = find_bottle_points(mask)
    bottle_center = find_center(bottle_points)
    narrow_end = find_farthest_point(bottle_points, bottle_center)
    wide_end = find_farthest_point(bottle_points, narrow_end)

    red_points = find_red_points(image, bottle_points)
    red_at_narrow_end = find_points_at_end(red_points, narrow_end, wide_end)
    red_at_wide_end = find_points_at_end(red_points, wide_end, narrow_end)

    enough_red_at_narrow_end = len(red_at_narrow_end) >= 5
    enough_red_at_wide_end = len(red_at_wide_end) >= 5
    more_red_at_narrow_end = len(red_at_narrow_end) > len(red_at_wide_end)
    cap_is_at_narrow_end = enough_red_at_narrow_end and more_red_at_narrow_end

    if cap_is_at_narrow_end:
        cap_center = find_center(red_at_narrow_end)
    elif enough_red_at_wide_end:
        cap_center = find_center(red_at_wide_end)
    else:
        points_at_narrow_end = find_points_at_end(bottle_points, narrow_end, wide_end)
        cap_center = find_center(points_at_narrow_end)

    sure = cap_is_at_narrow_end and not touches_edge(mask)

    # return format to be worked with, most important API detail
    return {'found': True, 'bottle_center': bottle_center, "cap_center": cap_center, "sure": sure}


def find_mask(image):
    results = model(image, retina_masks=True, verbose=False)
    result = results[0]

    number_of_bottles = len(result.boxes)
    if number_of_bottles == 0:
        return None

    masks = result.masks.data
    best_mask = masks[0]
    return best_mask.cpu().numpy()


def find_bottle_points(mask):
    left, top, width, height = cv2.boundingRect(mask)
    right = left + width
    bottom = top + height

    longer_side = max(width, height)
    step = longer_side // 100 + 1

    bottle_points = []
    for y in range(top, bottom, step):
        for x in range(left, right, step):
            point_is_on_bottle = mask[y][x] > 0
            if point_is_on_bottle:
                point = (x, y)
                bottle_points.append(point)

    return bottle_points


def find_center(points):
    sum_x = 0
    sum_y = 0
    for point in points:
        sum_x = sum_x + point[0]
        sum_y = sum_y + point[1]

    center_x = round(sum_x / len(points))
    center_y = round(sum_y / len(points))
    return (center_x, center_y)


def find_farthest_point(points, from_point):
    farthest_point = points[0]
    farthest_distance = 0
    for point in points:
        distance = find_distance(point, from_point)
        if distance > farthest_distance:
            farthest_point = point
            farthest_distance = distance
    return farthest_point


def find_distance(point_1, point_2):
    dx = point_1[0] - point_2[0]
    dy = point_1[1] - point_2[1]
    return math.sqrt(dx * dx + dy * dy)


def find_red_points(image, points):
    red_points = []
    for point in points:
        x = point[0]
        y = point[1]
        pixel = image[y][x]
        if is_red(pixel):
            red_points.append(point)
    return red_points


def is_red(pixel):
    blue = pixel[0]
    green = pixel[1]
    red = pixel[2]

    bright_enough = red >= 80
    green_is_low = green < 0.6 * red
    blue_is_low = blue < 0.6 * red

    return bright_enough and green_is_low and blue_is_low


def find_points_at_end(points, end, other_end):
    bottle_length = find_distance(end, other_end)

    points_at_end = []
    for point in points:
        distance_from_other_end = find_distance(point, other_end)
        point_is_at_end = distance_from_other_end > 0.85 * bottle_length

        if point_is_at_end:
            points_at_end.append(point)
    return points_at_end


def touches_edge(mask):
    left, top, width, height = cv2.boundingRect(mask)
    right = left + width
    bottom = top + height
    image_height = mask.shape[0]
    image_width = mask.shape[1]

    touches_left = left == 0
    touches_top = top == 0
    touches_right = right == image_width
    touches_bottom = bottom == image_height

    return touches_left or touches_top or touches_right or touches_bottom


def draw_bottle(image, bottle):
    if not bottle["found"]:
        return

    bottle_center = bottle["bottle_center"]
    cap_center = bottle["cap_center"]

    green = (0, 255, 0)
    red = (0, 0, 255)
    blue = (255, 0, 0)
    yellow = (0, 255, 255)

    if bottle["sure"]:
        line_color = green
    else:
        line_color = red

    cv2.line(image, bottle_center, cap_center, line_color, 4)
    cv2.circle(image, bottle_center, 10, blue, -1)
    cv2.circle(image, cap_center, 10, yellow, -1)
    # cv2.imshow("debug", image)


def draw_on_image(image_file, result_file):
    image = cv2.imread(image_file)
    bottle = find_bottle(image)
    draw_bottle(image, bottle)
    cv2.imwrite(result_file, image)
    return bottle


def draw_on_video(video_file, result_file):
    video = cv2.VideoCapture(video_file)
    ok, frame = video.read()

    fps = video.get(cv2.CAP_PROP_FPS)
    height = frame.shape[0]
    width = frame.shape[1]
    size = (width, height)
    video_type = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    result_video = cv2.VideoWriter(result_file, video_type, fps, size)

    bottles = []
    while ok:
        bottle = find_bottle(frame)
        draw_bottle(frame, bottle)
        result_video.write(frame)
        bottles.append(bottle)
        ok, frame = video.read()

    video.release()
    result_video.release()
    return bottles


def draw_on_camera(camera_number):
    camera = cv2.VideoCapture(camera_number)
    ok, frame = camera.read()

    while ok:
        bottle = find_bottle(frame)
        draw_bottle(frame, bottle)
        cv2.imshow("press any key to stop", frame)

        key = cv2.waitKey(1)
        key_was_pressed = key != -1
        if key_was_pressed:
            break
        ok, frame = camera.read()

    camera.release()
    cv2.destroyAllWindows()