from time import time as timer
import numpy as np
import cv2
from darkflow.net.build import TFNet
import math

if __name__ == '__main__':
    img_origin = cv2.imread('bin/yrm_origin.jpg')  # 无弹幕
    img_dm = cv2.imread('bin/yrm_dm.jpg')  # 有弹幕
    if img_origin is None or img_dm is None:
        print('img_origin is None or img_dm is None')
        exit(0)

    options = {
        'model': 'cfg/yolo.cfg',
        'load': 'bin/yolo.weights',
        'threshold': 0.5,
        'gpu': 0.7,
        # 'imgdir': 'sample_img/wa-yolo/',
    }
    tfnet = TFNet(options)

    boxes = tfnet.return_predict(img_origin)

    if len(boxes) == 0:
        print('there is no obj')
        exit(0)

    for result in boxes:
        if result['label'] != 'person':
            continue
        # 若此box为人
        (tx, ty) = (result['topleft']['x'], result['topleft']['y'])
        (bx, by) = (result['bottomright']['x'], result['bottomright']['y'])
        img_dm[ty:by, tx:bx] = img_origin[ty:by, tx:bx] // 4 * 3 + img_dm[ty:by, tx:bx] // 4

    while (1):
        cv2.imshow('image', img_dm)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q键退出
            break
    cv2.destroyAllWindows()
