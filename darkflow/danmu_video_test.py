from time import time as timer
import numpy as np
import cv2
from darkflow.net.build import TFNet
# import math
# from multiprocessing.pool import ThreadPool
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_shape(danmu, origin):
    '''
    检查两个视频的大小是否吻合，返回shape
    :param danmu:
    :param origin:
    :return: shape
    '''
    # 预读一帧获取视频大小
    _, frame_ori = origin.read()
    _, frame_dan = danmu.read()
    assert frame_dan.shape == frame_ori.shape, \
        "The height and width are not the same! \n {} with {}".format(frame_ori.shape, frame_dan.shape)
    return frame_dan.shape


def get_frame(danmu, origin):
    '''
    检查帧率
    :param danmu:
    :param origin:
    :return:
    '''
    if origin.get(cv2.CAP_PROP_FPS) != danmu.get(cv2.CAP_PROP_FPS):
        logger.warning('frame not same!')
        logger.warning('origin:{}'.format(origin.get(cv2.CAP_PROP_FPS)))
        logger.warning('danmu:{}'.format(danmu.get(cv2.CAP_PROP_FPS)))
    return origin.get(cv2.CAP_PROP_FPS)


def camera(tfnet, output, VIDEO_ORIGIN, VIDEO_DANMU, keep = -1):
    '''

    :param tfnet: YOLOV2网络实体
    :param output: 输出位置
    :param VIDEO_ORIGIN: 原始视频
    :param VIDEO_DANMU: 带弹幕的视频
    :param keep: 指定持续时间，单位为秒。-1为完整转换
    :return:
    '''

    # opencv打开视频
    origin = cv2.VideoCapture(VIDEO_ORIGIN)
    danmu = cv2.VideoCapture(VIDEO_DANMU)
    if not origin or not danmu:
        logger.warning('cant read the video~')
        exit(0)

    # 检查两个视频的大小是否吻合，返回shape
    height, width, _ = get_shape(danmu, origin)

    # 检查帧率是否吻合
    fps = get_frame(danmu, origin)

    if tfnet.FLAGS.saveVideo:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = round(danmu.get(cv2.CAP_PROP_FPS))
        videoWriter = cv2.VideoWriter(
            output, fourcc, fps, (width, height))

    temp_boxes = list()
    elapsed = int()
    start = timer()
    # Loop through frames
    while origin.isOpened():
        elapsed += 1
        _, frame_ori = origin.read()
        _, frame_dan = danmu.read()
        if frame_ori is None or frame_dan is None:
            logger.info('\nEnd of Video')
            break

        boxes = tfnet.return_predict(frame_ori)

        if len(boxes) == 0:
            continue

        # trick：因为yolo不稳定，有些frame会识别不出box，所以我们给box一个延时
        has_person = False
        for result in boxes:
            if result['label'] == 'person':
                has_person = True
        if not has_person:
            boxes = temp_boxes
        else:
            temp_boxes = boxes

        for result in boxes:
            if result['label'] != 'person':
                continue
            (tx, ty) = (result['topleft']['x'], result['topleft']['y'])
            (bx, by) = (result['bottomright']['x'], result['bottomright']['y'])
            frame_dan[ty:by, tx:bx] = frame_ori[ty:by, tx:bx]  # * 3 + frame_dan[ty:by, tx:bx] // 4

        if tfnet.FLAGS.saveVideo:
            videoWriter.write(frame_dan)

        if elapsed % 5 == 0:
            sys.stdout.write('\r')
            sys.stdout.write('{0:3.3f} FPS'.format(
                elapsed / (timer() - start)))
            sys.stdout.flush()

        # 每15帧清空一下延时box，以免无人物的画面中一直持续出现延时box
        if elapsed % 15 == 0:
            temp_boxes = list()
        if elapsed / fps == keep:
            logger.info('Made a video of {} seconds.'.format(keep))
            break

    sys.stdout.write('\n')
    if tfnet.FLAGS.saveVideo:
        videoWriter.release()
    origin.release()
    danmu.release()


if __name__ == "__main__":
    options = {
        'model': 'cfg/yolo.cfg',
        'load': 'bin/yolo.weights',
        'threshold': 0.30,
        'gpu': 0.7,
        'demo': 'sample_video/gakki_cut2_origin.flv',
        'saveVideo': True,
    }
    output = 'bin/' + options['demo'].split('/')[1].split('.')[0] + '_' + str(options['threshold']) + '_' + \
             options['model'].split('/')[1].split('.')[0] + '.avi'
    tfnet = TFNet(options)
    VIDEO_ORIGIN = tfnet.FLAGS.demo
    VIDEO_DANMU = 'sample_video/gakki_cut2_danmu.flv'
    KEEP = 30
    camera(tfnet, output, VIDEO_ORIGIN, VIDEO_DANMU, KEEP)
