from pycocotools.coco import COCO
import numpy as np
import cv2

class unified_coco:
        
    def __init__(self, anno_val = '../annotations/instances_val2014.json', anno_train = '../annotations/instances_train2014.json'):
        self.coco_train = COCO(anno_train)
        self.coco_val = COCO(anno_val)
        # self.category_dict = {1: 'person', 4: 'motorbike', 6: 'bus', 7: 'train', 8: 'truck', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 24: 'zebra', 25: 'giraffe', 46: 'glass', 74: 'mouse', 76: 'keyboard', 89: 'hair drier'}
        self.category_id = {'person': 1, 'motorbike': 4, 'bus': 6, 'train': 7, 'truck': 8, 'bird': 16, 'cat': 17, 'dog': 18, 'horse': 19, 'sheep': 20, 'cow': 21, 'elephant': 22, 'zebra': 24, 'giraffe': 25, 'glass': 46, 'mouse': 74, 'keyboard': 76, 'hair drier': 89}   
    
    def get_category_id(self, category_name):
        return self.category_id[category_name]
        
    def get_mask_of_category(self, category_name, img_name): 
        for coco in [self.coco_train, self.coco_val]:
            try:
                # coco = self.coco_train
                # print(category_name, img_name, end=' ')
                category_id = self.get_category_id(category_name)
                ann_ids = coco.getAnnIds(imgIds=int(img_name[-10:-4]), catIds=category_id)
                # print(category_id, ann_ids)
                anns = coco.loadAnns(ann_ids)
                mask = np.zeros((1024, 1280))
                if len(anns) == 0:
                    continue
                for ann in anns:
                    mask += cv2.resize(coco.annToMask(ann), (1280, 1024))
                mask[mask > 0] = 1
                return mask
            except:
                continue
        
    
    def get_mask_of_img(self, img_name): 
        for coco in [self.coco_train, self.coco_val]:
            try:
                ann_ids = coco.getAnnIds(imgIds=int(img_name[-10:-4]))
                # print(category_id, ann_ids)
                anns = coco.loadAnns(ann_ids)
                mask = np.zeros((1024, 1280))
                if len(anns) == 0:
                    # print(f'Error: {img_name} not found in {coco}')
                    continue
                for ann in anns:
                    mask += cv2.resize(coco.annToMask(ann), (1280, 1024))
                mask[mask > 0] = 1
                return mask
            except:
                # print(f'Error: {img_name}')
                continue
        
        
    def get_mask_from_anno(self, anno): 
        # is_train = 'train' in img_name
        # coco = self.coco_train if is_train else self.coco_val
        try:
            mask = self.coco_train.annToMask(anno)
        except:
            mask = self.coco_val.annToMask(anno)
        mask = cv2.resize(mask, (1280, 1024))
        return mask
        