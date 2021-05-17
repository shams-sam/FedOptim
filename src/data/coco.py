import os

from PIL import Image
import torch
from torchvision.datasets import CocoDetection

import common.config as cfg

class Coco(CocoDetection):
    def __init__(self, **kwargs):
        super(Coco, self).__init__(**kwargs)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        im_resize = cfg.im_size['coco']
        ann_loaded = coco.loadAnns(ann_ids)
        if len(ann_loaded):
            ann = ann_loaded[0]
        else:
            ann = False
        if ann:
            target = coco.annToMask(ann)
        else:
            target = torch.zeros(im_resize, im_resize)
        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
