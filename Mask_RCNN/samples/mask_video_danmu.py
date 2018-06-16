import logging
import os
import sys
from time import time as timer
import cv2
import numpy as np

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


def apply_mask(origin, danmu, mask):
    '''
    将origin图片的mask区域放到danmu图片之上
    :param origin:
    :param danmu:
    :param mask:
    :return:
    '''
    for c in range(3):
        danmu[:, :, c] = np.where(mask == 1,
                                  origin[:, :, c],
                                  danmu[:, :, c])
    return danmu


def display_instances(origin, danmu, boxes, masks, ids, scores):
    '''
    返回处理蒙版弹幕后的图片
    :param origin: 原始图片
    :param danmu: 带弹幕的图片
    :param boxes: 框, [num_instances, ... ...]
    :param masks: 掩膜，[h, w, num_instances]
    :param ids: 各个mask所属的class id, [num_instances]
    :param scores:  置信度
    :return:
    '''
    n_instances = boxes.shape[0]
    h, w, _ = origin.shape

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i in range(n_instances):
        if not np.any(boxes[i]):
            continue
        if ids[i] != 1:
        	# 1：person
            continue
        mask = masks[:, :, i]

        danmu = apply_mask(origin, danmu, mask)

    return danmu


def process(model, options):
    '''
    处理视频，输出处理后的蒙版弹幕视频
    :param model: Mask-RCNN模型
    :param options: 配置dict
    :return:
    '''
    if not os.path.exists(options.get('origin')):
        logger.warning('Origin video is not exists!')
        exit(0)

    if not os.path.exists(options.get('danmu')):
        logger.warning('Danmu video is not exists!')
        exit(0)

    # opencv打开视频
    origin = cv2.VideoCapture(options.get('origin'))
    danmu = cv2.VideoCapture(options.get('danmu'))

    saveVideo = options.get('saveVideo')

    # 检查两个视频的大小是否吻合，返回shape
    height, width, _ = get_shape(danmu, origin)

    # 检查帧率是否吻合
    fps = get_frame(danmu, origin)

    if saveVideo:
        if not os.path.exists(options.get('output').split('/')[0]):
            os.makedirs(options.get('output').split('/')[0])
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = round(danmu.get(cv2.CAP_PROP_FPS))
        videoWriter = cv2.VideoWriter(
            options.get('output'), fourcc, fps, (width, height))

    elapsed = int()  # 记录处理了多少帧
    start = timer()
    stop = fps * options.get('keep')

    # Loop through frames
    while origin.isOpened():
        elapsed += 1
        _, frame_ori = origin.read()
        _, frame_dan = danmu.read()
        if frame_ori is None or frame_dan is None:
            logger.info('\nEnd of Video')
            break
        results = model.detect([frame_ori], verbose=0)
        r = results[0]
        frame = display_instances(frame_ori, frame_dan, r['rois'], r['masks'], r['class_ids'], r['scores'])

        if saveVideo:
            videoWriter.write(frame)

        # 输出fps
        if elapsed % 5 == 0:
            sys.stdout.write('\r')
            sys.stdout.write('{0:3.3f} FPS'.format(
                elapsed / (timer() - start)))
            sys.stdout.flush()

        if elapsed == stop:
            logger.info('Made a video of {} seconds.'.format(options.get('keep')))
            break

    sys.stdout.write('\n')
    if saveVideo:
        videoWriter.release()
    origin.release()
    danmu.release()


if __name__ == '__main__':

    ROOT_DIR = os.path.abspath("../")

    sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
    sys.path.append(os.path.join(ROOT_DIR, "mrcnn/"))  # To find local version

    import coco
    import utils
    import model as modellib

    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    options = {
        'threshold': 0.5,  # 阈值
        'origin': 'sample_video/gakki_cut2_origin.flv',  # 原始视频
        'danmu': 'sample_video/gakki_cut2_danmu.flv',  # 弹幕视频
        'saveVideo': True,
        'output': 'output/gakki_cut2_2.avi',
        'keep': -1,  # 处理多少秒，-1代表处理完整视频
    }

    class InferenceConfig(coco.CocoConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = options.get('threshold')

    config = InferenceConfig()
    config.display()

    model = modellib.MaskRCNN(
        mode="inference", model_dir=MODEL_DIR, config=config
    )
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    # 处理视频
    process(model, options)