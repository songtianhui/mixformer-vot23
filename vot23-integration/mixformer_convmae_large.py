import os
import sys
import cv2
import numpy as np

import vot23 as vot
from lib.test.tracker.mixformer_convmae_online import MixFormerOnline
import lib.test.parameter.mixformer_convmae_online as vot_param
from external.AR.pytracking.ARcm_seg import ARcm_seg
from external.AR.pytracking.vot20_utils import *


class TrackerAR(object):
    def __init__(self, tracker,
                 refine_model_name='ARcm_coco_seg', threshold=0.6):
        self.tracker = tracker

        '''Alpha-Refine'''
        self.thres = threshold
        project_path = '/data3/sth/projects/MixFormer-vot23/external/AR/'
        refine_root = os.path.join(project_path, 'ltr/checkpoints/')
        refine_path = os.path.join(refine_root, refine_model_name)
        self.alpha = ARcm_seg(refine_path, input_sz=384)

    def initialize(self, image, mask):
        region = rect_from_mask(mask)
        init_info = {'init_bbox': region}
        self.tracker.initialize(image, init_info)

        self.H, self.W, _ = image.shape
        gt_bbox_np = np.array(region).astype(np.float32)
        self.alpha.initialize(image, np.array(gt_bbox_np))

    def track(self, image):
        '''base tracker'''
        out_dict = self.tracker.track(image)
        pred_bbox = out_dict['target_bbox']
        '''mask report'''
        pred_mask, search, search_mask = self.alpha.get_mask(image, np.array(pred_bbox), vis=True)
        final_mask = (pred_mask > self.thres).astype(np.uint8)
        return final_mask


def make_full_size(x, output_sz):
    '''
    zero-pad input x (right and down) to match output_sz
    x: numpy array e.g., binary mask
    output_sz: size of the output [width, height]
    '''
    if x.shape[0] == output_sz[1] and x.shape[1] == output_sz[0]:
        return x
    pad_x = output_sz[0] - x.shape[1]
    if pad_x < 0:
        x = x[:, :x.shape[1] + pad_x]
        # padding has to be set to zero, otherwise pad function fails
        pad_x = 0
    pad_y = output_sz[1] - x.shape[0]
    if pad_y < 0:
        x = x[:x.shape[0] + pad_y, :]
        # padding has to be set to zero, otherwise pad function fails
        pad_y = 0
    return np.pad(x, ((0, pad_y), (0, pad_x)), 'constant', constant_values=0)


handle = vot.VOT("mask", multiobject=True)
objects = handle.objects()
imagefile = handle.frame()

if not imagefile:
    sys.exit(0)

image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)
masks = [make_full_size(mask, (image.shape[1], image.shape[0])) for mask in objects]

import time

trackers = []
for i, mask in enumerate(masks):
    s = time.time()
    refine_model_name = 'ARcm_seg'
    params = vot_param.parameters('baseline_large', model='mixformer_convmae_large_online.pth.tar')
    mixformer = MixFormerOnline(params, "VOTS23")
    tracker = TrackerAR(mixformer, refine_model_name)
    tracker.initialize(image, mask)
    trackers.append(tracker)
    e = time.time()
    print(e - s)

print(trackers)

while True:
    imagefile = handle.frame()
    if not imagefile:
        break

    image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)

    states = []
    states = [tracker.track(image) for tracker in trackers]
    handle.report(states)
