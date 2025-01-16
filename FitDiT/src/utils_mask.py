import numpy as np
import cv2
from PIL import Image, ImageDraw
from skimage.measure import label, regionprops

def remove_small(binary_mask, min_area=10000):
    binary_mask = binary_mask.astype(np.uint8)

    label_image = label(binary_mask)
    regions = regionprops(label_image)
    small_white_regions = [region for region in regions if region.area < min_area]

    for region in small_white_regions:
        min_row, min_col, max_row, max_col = region.bbox
        binary_mask[min_row:max_row, min_col:max_col] = 0
    return binary_mask


def get_img_agnostic_upper_rectangle(im_parse, pose_data, offset_top, offset_bottom, offset_left, offset_right):
    foot = pose_data[18:24]

    faces = pose_data[24:92]

    hands1 = pose_data[92:113]
    hands2 = pose_data[113:]
    body = pose_data[:18]
    parse_array = np.array(im_parse)
    parse_upper_all = ((parse_array == 4).astype(np.float32) +
                    (parse_array == 7).astype(np.float32) +
                    (parse_array == 14).astype(np.float32) +
                    (parse_array == 15).astype(np.float32)
                    )
    parse_upper = ((parse_array == 4).astype(np.float32) +
                     (parse_array == 7).astype(np.float32)
                    )
    parse_head = ((parse_array == 3).astype(np.float32) +
                    (parse_array == 1).astype(np.float32) + (parse_array == 11).astype(np.float32))
    parse_fixed = (parse_array == 16).astype(np.float32)


    agnostic = Image.new(mode='L',size=(parse_array.shape[1], parse_array.shape[0]), color=0)
    img_black = Image.new(mode='L',size=(im_parse.shape[1], im_parse.shape[0]),color=0)

    gray_img = Image.new('L', size=(im_parse.shape[1], im_parse.shape[0]), color=128)
    agnostic_draw = ImageDraw.Draw(agnostic)

    parse_upper = np.uint8(parse_upper*255)
    parse_upper_all = np.uint8(parse_upper_all*255)

    contours_all, _ = cv2.findContours(parse_upper_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(parse_upper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cloth_exist = False
    try:
        # Initialize variables to store the extreme points
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= 100:
                x, y, w, h = cv2.boundingRect(contour)
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x + w)
                max_y = max(max_y, y + h)
                cloth_exist = True
        _x1, _y1, _x2, _y2 = min_x, min_y, max_x, max_y
    except:
        cloth_exist = False

    x1 = [body[i, 0] for i in [2, 3, 4] if body[i, 0] != 0]+\
        [hands2[i, 0] for i in [5,9,13] if hands2[i, 0] != 0]+\
        [hands1[i, 0] for i in [5,9,13] if hands1[i, 0] != 0]
    if x1:
        x1 = np.min(x1)
    else:
        x1 = 0
    y1 = [body[i, 1] for i in [2, 5] if body[i, 1] != 0]+\
          [hands2[i, 1] for i in [5,9,13] if hands2[i, 1] != 0]+\
            [hands1[i, 1] for i in [5,9,13] if hands1[i, 1] != 0]
    if y1:
        y1 = np.min(y1)
    else:
        y1 = 0
    x2 = [body[i, 0] for i in [5, 6, 7] if body[i, 0] != 0]+\
        [hands1[i, 0] for i in [5,9,13] if hands1[i, 0] != 0]+\
        [hands2[i, 0] for i in [5,9,13] if hands2[i, 0] != 0]
    if x2:
        x2 = np.max(x2)
    else:
        x2 = parse_array.shape[1]
    y2 = [body[i, 1] for i in [8, 11] if body[i, 1] != 0]+\
        [hands2[i, 1] for i in [5,9,13] if hands2[i, 1] != 0]+\
        [hands1[i, 1] for i in [5,9,13] if hands1[i, 1] != 0]
    if y2:
        y2 = np.max(y2)
    else:
        y2 = parse_array.shape[0]

    pad_y1 = 20
    pad_y2 = 10
    pad_x1 = pad_x2 = 25

    if cloth_exist:
        x1 = min(x1, _x1)
        x2 = max(x2, _x2)
        y1 = min(y1, _y1)
        if y2>_y2:
            pad_y2 = 0
        y2 = max(y2, _y2)

    y_face = [faces[i, 1] for i in [5, 11] if faces[i, 1] != 0]

    if y_face:
        y_face = np.mean(y_face)
        if y_face<y1:
            pad_y1 = 0
        y1 = min(y1, y_face)


    agnostic_draw.rectangle((max(0, x1-pad_x1+offset_left), max(0, y1-pad_y1+offset_top), min(x2+pad_x2+offset_right, parse_array.shape[1]), min(y2+pad_y2+offset_bottom, parse_array.shape[0])), 'gray', 'gray')
    agnostic.paste(img_black, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
    # agnostic.paste(img_black, None, Image.fromarray(np.uint8(parse_fixed * 255), 'L'))

    mask_gray = agnostic.copy()
    mask = agnostic.point(lambda p: p == 128 and 255)
  
    return mask, mask_gray


def get_img_agnostic_lower_rectangle(im_parse, pose_data, offset_top, offset_bottom, offset_left, offset_right):
    foot1 = pose_data[18:21] # right
    foot2 = pose_data[21:24] # left

    faces = pose_data[24:92]

    hands1 = pose_data[92:113]
    hands2 = pose_data[113:]
    body = pose_data[:18]
    parse_array = np.array(im_parse)

    parse_upper = ((parse_array == 4).astype(np.float32) +
                   (parse_array == 14).astype(np.float32) +
                   (parse_array == 15).astype(np.float32)
                    )
    parse_lower = ((parse_array == 5).astype(np.float32) +
                    (parse_array == 6).astype(np.float32) +
                    (parse_array == 8).astype(np.float32))
    parse_lower = np.uint8(parse_lower*255)
    parse_lower = remove_small(parse_lower, min_area=1000*(parse_lower.shape[0]*parse_lower.shape[1]/1024/768))
    parse_head = ((parse_array == 3).astype(np.float32) +
                    (parse_array == 1).astype(np.float32) + (parse_array == 11).astype(np.float32))
    parse_shoes = ((parse_array == 9).astype(np.float32) + (parse_array == 10).astype(np.float32))
    parse_fixed = (parse_array == 16).astype(np.float32)


    agnostic = Image.new(mode='L',size=(parse_array.shape[1], parse_array.shape[0]), color=0)
    img_black = Image.new(mode='L',size=(im_parse.shape[1], im_parse.shape[0]),color=0)

    gray_img = Image.new('L', size=(im_parse.shape[1], im_parse.shape[0]), color=128)
    agnostic_draw = ImageDraw.Draw(agnostic)

    
    contours, _ = cv2.findContours(parse_lower, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cloth_exist = False
    try:
        largest_contour = max(contours, key=cv2.contourArea)
        _x1, _y1, _w, _h = cv2.boundingRect(largest_contour)
        _x2, _y2 = _w+_x1, _h+_y1
        cloth_exist = True
    except:
        cloth_exist = False

    if not cloth_exist:
        parse_lower = ((parse_array == 5).astype(np.float32) +
                    (parse_array == 6).astype(np.float32) +
                    (parse_array == 7).astype(np.float32) +
                    (parse_array == 8).astype(np.float32))
        parse_lower = np.uint8(parse_lower*255)
        contours, _ = cv2.findContours(parse_lower, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        try:
            largest_contour = max(contours, key=cv2.contourArea)
            _x1, _y1, _w, _h = cv2.boundingRect(largest_contour)
            _x2, _y2 = _w+_x1, _h+_y1
            all_cloth_exist = True
        except:
            all_cloth_exist = False

    x1 = [body[i, 0] for i in [8, 9, 10] if body[i, 0] != 0]+[foot2[i, 0] for i in [0, 1, 2] if foot2[i, 0] != 0]+\
        [body[i, 0] for i in [11, 12, 13] if body[i, 0] != 0]+[foot1[i, 0] for i in [0, 1, 2] if foot1[i, 0] != 0]
    if x1:
        x1 = np.min(x1)
    else:
        x1=0
    y1 = min(body[8, 1], body[11, 1])
    x2 = [body[i, 0] for i in [11, 12, 13] if body[i, 0] != 0]+[foot1[i, 0] for i in [0, 1, 2] if foot1[i, 0] != 0]+\
        [body[i, 0] for i in [8, 9, 10] if body[i, 0] != 0]+[foot2[i, 0] for i in [0, 1, 2] if foot2[i, 0] != 0]
    if x2:
        x2 = np.max(x2)
    else:
        x2 = parse_array.shape[1]
    y2 = [body[i, 1] for i in [10, 13] if body[i, 1] != 0]+[foot1[i, 1] for i in [2] if foot2[i, 1] != 0]+\
        [foot2[i, 1] for i in [2] if foot2[i, 1] != 0]
    if y2:
        y2 = np.max(y2)
    else:
        y2 = parse_array.shape[0]
    if cloth_exist:
        x1 = min(x1, _x1)
        x2 = max(x2, _x2)
        y1 = min(y1, _y1)
        y2 = max(y2, _y2)
    elif all_cloth_exist:
        x1 = min(x1, _x1)
        x2 = max(x2, _x2)

    pad_y1 = 5 if cloth_exist else 50
    pad_y2 = 10
    pad_x1 = pad_x2 = 30


    agnostic_draw.rectangle((max(0, x1-pad_x1+offset_left), max(0, y1-pad_y1+offset_top), min(x2+pad_x2+offset_right, parse_array.shape[1]), min(y2+pad_y2+offset_bottom, parse_array.shape[0])), 'gray', 'gray')
    agnostic.paste(img_black, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
    agnostic.paste(img_black, None, Image.fromarray(np.uint8(parse_fixed * 255), 'L'))

    mask_gray = agnostic.copy()
    mask = agnostic.point(lambda p: p == 128 and 255)
    return mask, mask_gray


def get_img_agnostic_dresses_rectangle(im_parse, pose_data, offset_top, offset_bottom, offset_left, offset_right):
    foot1 = pose_data[18:21] # right
    foot2 = pose_data[21:24] # left

    faces = pose_data[24:92]

    hands1 = pose_data[92:113]
    hands2 = pose_data[113:]
    body = pose_data[:18]
    parse_array = np.array(im_parse)
    parse_upper_all = ((parse_array == 4).astype(np.float32) +
                       (parse_array == 5).astype(np.float32) +
                       (parse_array == 6).astype(np.float32) +
                        (parse_array == 7).astype(np.float32) +
                        (parse_array == 14).astype(np.float32) +
                        (parse_array == 15).astype(np.float32)
                    )

    parse_head = ((parse_array == 3).astype(np.float32) +
                    (parse_array == 1).astype(np.float32) + (parse_array == 11).astype(np.float32))
    parse_shoes = ((parse_array == 9).astype(np.float32) + (parse_array == 10).astype(np.float32))
    parse_fixed = (parse_array == 16).astype(np.float32)


    agnostic = Image.new(mode='L',size=(parse_array.shape[1], parse_array.shape[0]), color=0)
    img_black = Image.new(mode='L',size=(im_parse.shape[1], im_parse.shape[0]),color=0)

    gray_img = Image.new('L', size=(im_parse.shape[1], im_parse.shape[0]), color=128)
    agnostic_draw = ImageDraw.Draw(agnostic)

    parse_upper_all = np.uint8(parse_upper_all*255)
    # 查找轮廓
    contours_all, _ = cv2.findContours(parse_upper_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 找到最大的轮廓
    cloth_exist = False
    try:
        largest_contour_all = max(contours_all, key=cv2.contourArea)
        _x1, _y1, _w, _h = cv2.boundingRect(largest_contour_all)
        _x2 = _x1+_w
        _y2 = _y1+_h
        cloth_exist = True
    except:
        cloth_exist = False

    x1 = [body[i, 0] for i in [2, 3, 4] if body[i, 0] != 0]+\
        [body[i, 0] for i in [8, 9, 10] if body[i, 0] != 0]+\
        [foot2[i, 0] for i in [0, 1, 2] if foot2[i, 0] != 0]+\
        [hands2[i, 0] for i in [5, 9, 13, 17] if hands2[i, 0] != 0]
    if x1:
        x1 = np.min(x1)
    else:
        x1 = 0
    y1 = [body[2, 1], body[5, 1]]+\
          [hands2[i, 1] for i in [5, 9, 13, 17] if hands2[i, 1] != 0]+\
            [hands1[i, 1] for i in [5, 9, 13, 17] if hands1[i, 1] != 0]
    if y1:
        y1 = np.min(y1)
    else:
        y1 = 0
    x2 = [body[i, 0] for i in [5, 6, 7] if body[i, 0] != 0]+\
        [body[i, 0] for i in [11, 12, 13] if body[i, 0] != 0]+\
            [foot1[i, 0] for i in [0, 1, 2] if foot1[i, 0] != 0]+\
            [hands1[i, 0] for i in [5, 9, 13, 17] if hands1[i, 0] != 0]
    if x2:
        x2 = np.max(x2)
    else:
        x2 = parse_array.shape[1]
    y2 = [body[i, 1] for i in [10, 13] if body[i, 1] != 0]+[foot1[i, 1] for i in [0, 1, 2] if foot1[i, 1] != 0]+\
        [foot2[i, 1] for i in [0, 1, 2] if foot2[i, 1] != 0]
    if y2:
        y2 = np.max(y2)
    else:
        y2 = parse_array.shape[0]

    if cloth_exist:
        x1 = min(x1, _x1)
        x2 = max(x2, _x2)
        y1 = min(y1, _y1)
        y2 = max(y2, _y2)


    pad_y1 = 20
    pad_y2 = 10
    pad_x1 = pad_x2 = 50

    y_face = [faces[i, 1] for i in [5, 11] if faces[i, 1] != 0]

    if y_face:
        y_face = np.mean(y_face)
        if y_face<y1:
            pad_y1 = 0
        y1 = min(y1, y_face)


    agnostic_draw.rectangle((max(0, x1-pad_x1+offset_left), max(0, y1-pad_y1+offset_top), min(x2+pad_x2+offset_right, parse_array.shape[1]), min(y2+pad_y2+offset_bottom, parse_array.shape[0])), 'gray', 'gray')
    agnostic.paste(img_black, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
    agnostic.paste(img_black, None, Image.fromarray(np.uint8(parse_shoes * 255), 'L'))
    agnostic.paste(img_black, None, Image.fromarray(np.uint8(parse_fixed * 255), 'L'))

    mask_gray = agnostic.copy()
    mask = agnostic.point(lambda p: p == 128 and 255)
    return mask, mask_gray

def get_mask_location(category, model_parse, pose_data, width,height, \
                        offset_top, offset_bottom, offset_left, offset_right):
    im_parse = model_parse.resize((width, height), Image.NEAREST)
    parse_array = np.array(im_parse)

    if category== "Upper-body":
        mask, mask_gray = get_img_agnostic_upper_rectangle(parse_array, pose_data, offset_top, offset_bottom, offset_left, offset_right)
        return mask, mask_gray
    elif category=="Dresses":
        mask, mask_gray = get_img_agnostic_dresses_rectangle(parse_array, pose_data, offset_top, offset_bottom, offset_left, offset_right)
        return mask, mask_gray
    elif category=="Lower-body":
        mask, mask_gray = get_img_agnostic_lower_rectangle(parse_array, pose_data, offset_top, offset_bottom, offset_left, offset_right)
        return mask, mask_gray
