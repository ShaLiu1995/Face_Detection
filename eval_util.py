import os
import cv2
import json
from scipy.misc import imread, imsave, imresize


with open('bbx_val_dict.json') as json_data:
    bbx_val_dict = json.load(json_data)

    
def evaluate_test_set(lo, hi, bbx, image_dir):
    difficulty = image_dir.split('_')[-1]
    NEW_IMG_DIR = 'test_results_{}'.format(difficulty)
    if not os.path.isdir(NEW_IMG_DIR):
        os.mkdir(NEW_IMG_DIR)
        
    file_list = os.listdir(image_dir)
    for i in range(lo, hi):
        file = file_list[i]
        old_path = os.path.join(image_dir, file)
        new_path = os.path.join(NEW_IMG_DIR, file)
        draw_result(old_path, new_path, bbx[i - lo])
    print('Drawing output bounding box for {} completed'.format(image_dir))


def draw_bbx(img, pred_bbx, label_bbx):
    new_img = img.copy()
    cv2.rectangle(new_img, (pred_bbx[0], pred_bbx[1]),
                  (pred_bbx[0] + pred_bbx[2], pred_bbx[1] + pred_bbx[3]), (255, 0, 0), 2)
    cv2.rectangle(new_img, (label_bbx[0], label_bbx[1]),
                  (label_bbx[0] + label_bbx[2], label_bbx[1] + label_bbx[3]), (0, 255, 0), 2)
    return new_img.copy()


def draw_result(old_path, new_path, pred_bbx):
    img = imread(old_path, mode='RGB')
    h = img.shape[0]
    w = img.shape[1]
    if w != 224 or h != 224:
        img = imresize(img, (224, 224))
        
    filename = old_path.split(os.sep)[-1]
    label_bbx = bbx_val_dict[filename]
    img = draw_bbx(img, pred_bbx, label_bbx)
        
    if w != 224 or h != 224:
        img = imresize(img, (h, w))
    imsave(new_path, img)
    return img


def get_iou(bb1, bb2):

    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[0] + bb1[2], bb2[0] + bb2[2])
    y_bottom = min(bb1[1] + bb1[3], bb2[1] + bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = bb1[2] * bb1[3]
    bb2_area = bb2[2] * bb2[3]

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    try:
        assert iou >= 0.0
        assert iou <= 1.0
    except AssertionError:
        print(iou)
        print(bb1)
        print(bb2)
    return iou


def get_acc(ytest, ltest):
    assert ytest.shape[0] == ltest.shape[0]
    n = ytest.shape[0]
    count = 0
    for i in range(n):
        if get_iou(ytest[i], ltest[i]) > 0.5:
            count += 1
    return 1.0 * count / n
