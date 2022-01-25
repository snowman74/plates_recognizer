import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw, ImageFont


def final_detect(image_array, model_plates, model_chars, font_path):
    """
    image_array - NumPy array
    model_plates, model_chars - PyTorch models
    return - result image in numpy array format
    """
    results_plates = model_plates(image_array, size=1280)  # recognize image_array
    pred_list_plates = results_plates.pandas().xywhn[0]  # convert result to pandas DataFrame object
    pred_list_plates.drop(['confidence', 'name', 'class'], axis=1, inplace=True)  # drop any columns
    pred_list_plates = sort_detected_obj(pred_list_plates, mode='plates').values  # sort DataFrame by X_center
    if pred_list_plates.size:  # if the plate was recognized
        labels = get_labels_from_image(image_array, pred_list_plates, model_chars)  # get labels from detected plates
        result_image = get_image_with_bboxes(image_array,
                                             pred_list_plates,
                                             labels,
                                             font_path,
                                             plot_label=True)  # draw bboxes and labels on init image_array
    else:  # else result_image is initial image_array
        labels = []
        result_image = image_array
    return result_image, labels


def get_labels_from_image(image: object, bboxes: object, model_chars: object) -> object:
    """
    return - list labels (detected chars on plate)
    :param model_chars: PyTorch model for detecting chars on plate
    :param image: PIL Image object
    :type bboxes: list with coord bounding boxes
    """
    img = image.copy()
    img = Image.fromarray(img)
    labels = []
    for box in bboxes:
        x1, y1, x2, y2 = get_coord_bbox(img, box)
        img_crop = img.crop(box=(x1, y1, x2, y2))
        img_crop.thumbnail((160, 160), Image.ANTIALIAS)
        img_crop_rotate = rotate_image_v2(np.array(img_crop), show_line=True)
        result_chars = model_chars(img_crop_rotate, size=160).pandas().xywhn[0]
        if result_chars.size:
            label = get_labels_from_predict(result_chars)
        else:
            label = 'not recognized'
        labels.append(label)
    return labels


def get_font(font_path, image_weight, image_height):
    """

    :param font_path: str - path to font file
    :param image_weight: int - weight of detecting image
    :param image_height: int - height of detecting image
    :return: ImageFont object
    """
    thickness = (image_weight + image_height) // 80
    return ImageFont.truetype(font_path, thickness)



def get_image_with_bboxes(image_array, predict_list, labels, font_path, plot_label=True):
    """
    image_array - image numpy array
    predict_list - list with detected plates
    labels - list with recognized chars - fullnamed plates
    plot_label = bool param, if True - recognized labels will be draw on image (image_array)
    return - image in numpy array format
    """
    image = Image.fromarray(image_array)
    img_draw = ImageDraw.Draw(image)  # Create ImageDraw object to draw text and bboxes
    iw, ih = image.size
    font = get_font(font_path, iw, ih)
    for box, label in zip(predict_list, labels):
        x1, y1, x2, y2 = get_coord_bbox(image, box)
        img_draw.rectangle([x1, y1, x2, y2], outline='red')
        if plot_label:
            label_size = img_draw.textsize(label)
            if y1 - label_size[1] >= 0:
                text_origin = np.array([x1, y1 - label_size[1]])
            else:
                text_origin = np.array([x1, y1 + 1])

            img_draw.text(tuple(text_origin), label, fill=(0, 255, 0), font=font)
    return np.array(image, dtype='uint8')


def get_coord_bbox(image, bbox):
    """
    function transform X center Y center Weight Height presented by relative coords in X1 Y1 X2 Y2 absolute coords
    image - PIL Image object
    bbox - list with XYWH relative coords detected by the model
    return - list with X1 Y1 X2 Y2 absolute coords
    """
    iw, ih = image.size
    center_x, center_y, width_b, height_b = bbox
    center_x *= iw
    center_y *= ih
    width_b *= iw
    height_b *= ih
    # found coords left upper and right lower angles
    x1 = round(center_x - width_b / 2, 3)
    x2 = round(center_x + width_b / 2, 3)
    y1 = round(center_y - height_b / 2, 3)
    y2 = round(center_y + height_b / 2, 3)
    return [x1, y1, x2, y2]


def get_labels_from_predict(pred_list):
    """
    function returns the label from the plate recognized by the model by pandas DataFrame format
    pred_list - result model if pandas DataFrame object
    return - recognized label
    """
    return sort_detected_obj(pred_list, mode='chars')


def sort_detected_obj(predict_df, mode='chars'):
    """
    function for sort detected by model object
    predict_df - model recognition result in Pandas DataFrame format
    mode - string param. If mode = 'chars' function applied sort algorithm for chars in plate,
    if mode = 'plate' - function sort object (plates in image) by 'x_center' column
    return tuple (label, sorted DataFrame) if mode = 'chars'
    return sorted DataFrame if mode = 'plates'
    """
    if predict_df.size != 0:
        if mode == 'chars':
            if predict_df.ycenter.max() - predict_df.ycenter.min() > predict_df.height.max():
                second_line = predict_df[predict_df.ycenter > predict_df.ycenter.mean()].copy().sort_values(by='xcenter')
                first_line = predict_df[predict_df.ycenter < predict_df.ycenter.mean()].copy().sort_values(by='xcenter')
                sorted_df = pd.concat([first_line, second_line], axis=0)
            else:
                sorted_df = predict_df.sort_values(by='xcenter')
            return ''.join(list(sorted_df.name))
        if mode == 'plates':
            return predict_df.sort_values(by='xcenter')
    else:
        return predict_df

def rotate_image_v2(src, show_line=True):
    """
    function for rotate image by cv2.HoughLinesP method

    src - image in numpy array format
    show_line - bool, if True - on image will be draw line by rotate
    return - rotate image in numpy array format
    """
    image = src.copy()

    def rotate(image, degree):  # rotate image
        """
        function for rotate image by degree

        image - numpy array (image to rotate)
        degree - float - (degree, grad)
        return - rotate image
        """
        # add white border so as not to lose anything when rotate image
        init_w, init_h = image.shape[:2]
        outputImage = cv2.copyMakeBorder(image,
                                         int(init_w / 8),
                                         int(init_w / 8),
                                         int(init_h / 8),
                                         int(init_h / 8),
                                         cv2.BORDER_CONSTANT,
                                         value=(255, 255, 255))
        h, w = outputImage.shape[:2]
        RotateMatrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), degree, 1)
        rotate = cv2.warpAffine(outputImage,  # calc affine matrix
                                RotateMatrix,
                                (w, h),
                                borderValue=(255, 255, 255))
        return rotate

    def degree_trans(theta):  # find rotate angle
        """
        function for transform degree from rad to grad

        theta - float (degree in rad)
        return - float (degree in grad)
        """
        return theta / np.pi * 180

    def get_degree(coord):
        """
        function for get degree by coord line (x1, y1, x2, y2)

        coord - list containing x1, y1, x2, y2 line coordinates
        return - degree in rad format
        """
        x1, y1, x2, y2 = coord
        v1 = (x2 - x1, y2 - y1)
        v2 = (abs(x2 - x1), 0)
        cosAlpha = (v1[0] * v2[0] + v1[1] * v2[1]) / ((v1[0]) ** 2 + (v1[1]) ** 2) ** 0.5 / (
                    (v2[0]) ** 2 + (v2[1]) ** 2) ** 0.5
        return degree_trans(np.arccos(cosAlpha))

    def calc_coord(image, show_line):
        """
        function for get coordinates the longest line found using Hough transform

        image - image in numpy array format
        show_line - bool, if True - line will be draw on image,
        False - line will not be draw on image
        return - list, containing x1, y1, x2, y2 line coordinates
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dstImage = cv2.Canny(image, 300, 430, apertureSize=3, L2gradient=True)  # find the contours of the image
        lines = cv2.HoughLinesP(dstImage,
                                rho=1,
                                theta=np.pi / 180,
                                threshold=50,
                                minLineLength=10,
                                maxLineGap=15)
        # find the contours of the image
        d_lines = []
        coord_list = []
        try:
            for _ in lines:
                for line in lines:
                    for x1, y1, x2, y2 in line:
                        d = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5  # by coords calc lenghts of lines
                        coord_list.append([x1, y1, x2, y2])
                        d_lines.append(d)

                index_max_line = d_lines.index(
                    max(d_lines))  # looking for the longest line
                coord = coord_list[index_max_line]  # save its coordinates
        except TypeError:
            return [0, 0, 0, 0]
        if show_line:
            cv2.line(image, (coord[0], coord[1]), (coord[2], coord[3]), [255, 255, 255], 3, 8)
        return coord

    coord = calc_coord(image, show_line)
    try:
        degree = get_degree(coord)
    except ZeroDivisionError:
        degree = 0
    rotate_img = rotate(image, degree)
    return rotate_img
