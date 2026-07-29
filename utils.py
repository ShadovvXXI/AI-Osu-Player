import cv2

def draw_image_with_circle(image, center):
    # радиус круга в пикселях
    radius = 4

    # цвет в формате BGR (чёрный)
    color = (255, 255, 255)

    # залитый круг
    thickness = -1

    cv2.circle(image, center, radius, color, thickness)

    success = cv2.imwrite("result.jpg", image)
    if not success:
        raise IOError("Не удалось сохранить изображение")

def osu_cords_to_window_pos(cords, scale, offset):
    return int(cords[0] * scale + offset[0]), int(cords[1] * scale + offset[1])

def window_pos_to_train_pos(resolution, pos, width, height):
    return int(pos[0] / resolution[0] * width), int(pos[1] / resolution[1] * height)

def pred_pos_to_window_pos(resolution, pos, width, height):
    return int(pos[0] / width * resolution[0]), int(pos[1] / height * resolution[1])

def recover_coords(x, y, img_size): # TODO: объединить с to_train_pos
    x = ((x + 1) / 2) * img_size[0]
    y = ((y + 1) / 2) * img_size[1]
    return x, y

def approach_time_ms(ar):
    if ar < 5:
        return int(1800 - 120 * ar)
    else:
        return int(1200 - 150 * (ar - 5))