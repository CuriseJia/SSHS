import json
import os
import random
from collections import defaultdict
from pycocotools.coco import COCO
import numpy as np

DEPTH_FOLDER = '/home/yanhao/depth_train'

MONITOR_HALF_WIDTH = 0.542/2
MONITOR_HALF_HEIGHT = 0.305/2
MONITOR_DEPTH = 0.76

IMAGE_HALF_WIDTH = 1920/2
IMAGE_HALF_HEIGHT = 1080/2

cocoTrain = COCO('/home/yanhao/coco/annotations/instances_train2014.json')
cocoVal = COCO('/home/yanhao/coco/annotations/instances_val2014.json')

cats = ['dog', 'person', 'bird', 'motorcycle', 'keyboard', 'cat', 'cow', 'horse', 'boat', 'elephant', 'train', 'airplane']
cat_ids = cocoTrain.getCatIds(catNms=cats)

id_name = {cat_ids[i]: cats[i] for i in range(len(cats))}
name_id = {cats[i]: cat_ids[i] for i in range(len(cats))}

with open('./config1_train.json', 'r') as file:
    train_data = json.load(file)

with open('./config1_test.json', 'r') as file:
    test_data = json.load(file)

final_train = []
final_val = []

categories_train = defaultdict(list)
categories_test = defaultdict(list)


with open('./all_filtered_files_train.txt', 'r') as file:
    for line in file:
        line = line.strip()
        if line.startswith('#') or not line:
            continue
        
        parts = line.split('/')
        if len(parts) == 2:
            cat, audio = parts
            categories_train[cat].append(audio)

with open('./all_filtered_files_test.txt', 'r') as file:
    for line in file:
        line = line.strip()
        if line.startswith('#') or not line:
            continue
        
        parts = line.split('/')
        if len(parts) == 2:
            cat, audio = parts
            categories_test[cat].append(audio)

for data in train_data:
    image_id = data['image_id']
    imgId = int(image_id.split('_')[-1].split('.')[0])
    annIds = cocoTrain.getAnnIds(imgIds=imgId)
    anns = cocoTrain.loadAnns(annIds)

    img = cocoTrain.loadImgs(imgId)[0]
    curWidth = img['width']
    curHeight = img['height']
    widthScale = 1920/curWidth
    heightScale = 1080/curHeight

    cur_cat = data['category']
    if cur_cat == 'plane':
        cur_cat = 'airplane'
    cur_id = name_id[cur_cat]
    storeAnns = []
    for ann in anns:
        if ann['category_id'] in cat_ids and ann['category_id']!=cur_id:
            storeAnns.append(ann)

    if not storeAnns:
        print(f'No valid distractors for {imgId} in training dataset')
        print(f"Category {cur_cat}, {data['object_size']}")
        continue

    distAnn = random.choice(storeAnns)
    distbbox_original = distAnn['bbox']
    distbbox = [distAnn['bbox'][0]*widthScale, distAnn['bbox'][1]*heightScale, distAnn['bbox'][2]*widthScale, distAnn['bbox'][3]*heightScale]

    distCat = id_name[distAnn['category_id']]
    if distCat == 'airplane':
        distCat = 'plane'

    data['dist_category'] = distCat
    data['dist_gt_box'] = distbbox

    filePath = data['image_id'].split('.')[0] + '.npy'
    depth_npy_path = os.path.join(DEPTH_FOLDER, filePath)
    if not os.path.exists(depth_npy_path):
        continue
    depth_npy = np.load(depth_npy_path)
    if depth_npy.shape!=(1024, 1280):
        raise Exception("Depth array size mismatch")
    
    center_x = distbbox_original[0] + distbbox_original[2] / 2
    center_y = distbbox_original[1] + distbbox_original[3] / 2
    
    resized_center_x = center_x/curWidth * 1280
    resized_center_y = center_y/curHeight * 1024

    flipped_depth_value = depth_npy[int(resized_center_y), int(resized_center_x)]
    depth_value = 10 - flipped_depth_value

    data['dist_point'] = [distbbox[0]+distbbox[2]/2, distbbox[1]+distbbox[3]/2, float(depth_value)]

    selected_audio = random.choice(categories_train[distCat])
    data['dist_audio'] = distCat + '/' + selected_audio

    # Edit coordinates:
    target_coords = data['point']
    dist_coords = data['dist_point']

    new_x = target_coords[0] - IMAGE_HALF_WIDTH
    new_y = IMAGE_HALF_HEIGHT - target_coords[1]

    unity_x = (new_x/IMAGE_HALF_WIDTH) * MONITOR_HALF_WIDTH
    unity_y = (new_y/IMAGE_HALF_HEIGHT) * MONITOR_HALF_HEIGHT
    unity_z = (target_coords[2] / 10) * 0.74 + MONITOR_DEPTH

    data['unity_point'] = [unity_x, unity_y, unity_z]

    new_x = dist_coords[0] - IMAGE_HALF_WIDTH
    new_y = IMAGE_HALF_HEIGHT - dist_coords[1]

    unity_x = (new_x/IMAGE_HALF_WIDTH) * MONITOR_HALF_WIDTH
    unity_y = (new_y/IMAGE_HALF_HEIGHT) * MONITOR_HALF_HEIGHT
    unity_z = (dist_coords[2] / 10) * 0.74 + MONITOR_DEPTH

    data['dist_unity_point'] = [unity_x, unity_y, unity_z]

    final_train.append(data)

for data in test_data:
    image_id = data['image_id']
    imgId = int(image_id.split('_')[-1].split('.')[0])
    annIds = cocoVal.getAnnIds(imgIds=imgId)
    anns = cocoVal.loadAnns(annIds)

    img = cocoVal.loadImgs(imgId)[0]
    curWidth = img['width']
    curHeight = img['height']
    widthScale = 1920/curWidth
    heightScale = 1080/curHeight

    cur_cat = data['category']
    if cur_cat == 'plane':
        cur_cat = 'airplane'
    cur_id = name_id[cur_cat]
    storeAnns = []
    for ann in anns:
        if ann['category_id'] in cat_ids and ann['category_id']!=cur_id:
            storeAnns.append(ann)

    if not storeAnns:
        print(f'No valid distractors for {imgId} in test dataset')
        print(f"Category {cur_cat}, {data['object_size']}")
        continue

    distAnn = random.choice(storeAnns)
    distbbox_original = distAnn['bbox']
    distbbox = [distAnn['bbox'][0]*widthScale, distAnn['bbox'][1]*heightScale, distAnn['bbox'][2]*widthScale, distAnn['bbox'][3]*heightScale]

    distCat = id_name[distAnn['category_id']]
    if distCat == 'airplane':
        distCat = 'plane'

    data['dist_category'] = distCat
    data['dist_gt_box'] = distbbox

    filePath = data['image_id'].split('.')[0] + '.npy'
    depth_npy_path = os.path.join(DEPTH_FOLDER, filePath)
    if not os.path.exists(depth_npy_path):
        continue
    depth_npy = np.load(depth_npy_path)
    if depth_npy.shape!=(1024, 1280):
        raise Exception("Depth array size mismatch")
    
    center_x = distbbox_original[0] + distbbox_original[2] / 2
    center_y = distbbox_original[1] + distbbox_original[3] / 2
    
    resized_center_x = center_x/curWidth * 1280
    resized_center_y = center_y/curHeight * 1024

    flipped_depth_value = depth_npy[int(resized_center_y), int(resized_center_x)]
    depth_value = 10 - flipped_depth_value

    data['dist_point'] = [distbbox[0]+distbbox[2]/2, distbbox[1]+distbbox[3]/2, float(depth_value)]

    selected_audio = random.choice(categories_test[distCat])
    data['dist_audio'] = distCat + '/' + selected_audio

    # Edit coordinates:
    target_coords = data['point']
    dist_coords = data['dist_point']

    new_x = target_coords[0] - IMAGE_HALF_WIDTH
    new_y = IMAGE_HALF_HEIGHT - target_coords[1]

    unity_x = (new_x/IMAGE_HALF_WIDTH) * MONITOR_HALF_WIDTH
    unity_y = (new_y/IMAGE_HALF_HEIGHT) * MONITOR_HALF_HEIGHT
    unity_z = (target_coords[2] / 10) * 0.74 + MONITOR_DEPTH

    data['unity_point'] = [unity_x, unity_y, unity_z]

    new_x = dist_coords[0] - IMAGE_HALF_WIDTH
    new_y = IMAGE_HALF_HEIGHT - dist_coords[1]

    unity_x = (new_x/IMAGE_HALF_WIDTH) * MONITOR_HALF_WIDTH
    unity_y = (new_y/IMAGE_HALF_HEIGHT) * MONITOR_HALF_HEIGHT
    unity_z = (dist_coords[2] / 10) * 0.74 + MONITOR_DEPTH

    data['dist_unity_point'] = [unity_x, unity_y, unity_z]

    final_val.append(data)

with open('./master_train.json', 'w') as file:
    json.dump(final_train, file)

with open('./master_test.json', 'w') as file:
    json.dump(final_val, file)
