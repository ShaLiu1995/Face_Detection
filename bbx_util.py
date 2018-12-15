import cv2

def resize_bbx(img, bbox):
    height = img.shape[0]
    width = img.shape[1]
    new_bbox = [0] * 4
    new_bbox[0] = int(224 * bbox[0] / width)
    new_bbox[1] = int(224 * bbox[1] / height)
    new_bbox[2] = int(224 * bbox[2] / width)
    new_bbox[3] = int(224 * bbox[3] / height)
    return new_bbox


def resize_bbx_inverse(img, bbox):
    height = img.shape[0]
    width = img.shape[1]
    new_bbox = [0] * 4
    new_bbox[0] = int(width * bbox[0] / 224)
    new_bbox[1] = int(height * bbox[1] / 224)
    new_bbox[2] = int(width * bbox[2] / 224)
    new_bbox[3] = int(height * bbox[3] / 224)
    return new_bbox


def draw_bbx(img, pred_bbx, label_bbx=None):
    new_img = img.copy()
    cv2.rectangle(new_img, (pred_bbx[0], pred_bbx[1]),
                  (pred_bbx[0] + pred_bbx[2], pred_bbx[1] + pred_bbx[3]), (255, 0, 0), 2)
    if label_bbx is not None:
        cv2.rectangle(new_img, (label_bbx[0], label_bbx[1]),
                      (label_bbx[0] + label_bbx[2], label_bbx[1] + label_bbx[3]), (0, 255, 0), 2)
    return new_img.copy()