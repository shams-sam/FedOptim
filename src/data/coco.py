import os

from PIL import Image
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
        ann = coco.loadAnns(ann_ids)[0]
        target = coco.annToMask(ann)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
