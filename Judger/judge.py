from utils import unified_coco
import argparse
from tqdm import tqdm
import json
import math

if __name__ == '__main__':
    coco = unified_coco()
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=int, default=1)
    parser.add_argument('-i', '--result', type=str, required=True)
    args = parser.parse_args()
    task = args.task

    with open(f'../dataset/cond{task}/config.json') as f:
        config = json.load(f)
    
    if task != 5:
        with open(f'../dataset/cond{task}/area_class.json') as f:
            area_class = json.load(f)
        
    with open(args.result) as f:
        result = json.load(f)

    total = [0, 0, 0]
    success = [0, 0, 0]
    success_v = [0, 0, 0]
    success_a = [0, 0, 0]
    expect_success = [0, 0, 0]
    expect_success_v = [0, 0, 0]
    fake_location = {}
    if task in [2, 3]:
        for anno in config:
            fake_location[anno['audio'][2:]] = anno['fake_location']
    
    if task in [1, 3, 4, 6]:

        for wav_name, predict_point in tqdm(result.items()):
            if type(predict_point) != tuple and type(predict_point) != list or len(predict_point) != 2:
                print(f'[error] {wav_name} predict_point is not tuple/list or length is not 2')
                exit(1)

            center_x, center_y = predict_point
            center_x, center_y = round(center_x), round(center_y)
            area_id, area_mask = area_class[wav_name][0], area_class[wav_name][1]
            area_mask = coco.get_mask_from_anno(area_mask)
            expect_success[area_id] += (area_mask > 0.5).sum() / (1024 * 1280)
                    
            if area_mask[center_x, center_y] > 0.5:
                success[area_id] += 1
            
            elif task == 6:
                category = wav_name.split('_')[-2]
                area_mask_all = coco.get_mask_of_category(category, wav_name.split('_')[-3] + '.jpg')
                if area_mask_all[center_x, center_y] > 0.5:
                    success_v[area_id] += 1
                expect_success_v[area_id] += (area_mask_all > 0.5).sum() / (1024 * 1280)
                
            elif task == 3:
                gt_y, gt_x = fake_location[wav_name][0], fake_location[wav_name][1]
                if math.sqrt((center_x - gt_x) ** 2 + (center_y - gt_y) ** 2) < 200:
                    success_a[area_id] += 1
                
                    
            total[area_id] += 1
            
    elif task == 2:   
            
        for wav_name, predict_point in tqdm(result.items()):
            if type(predict_point) != tuple and type(predict_point) != list or len(predict_point) != 2:
                print(f'[error] {wav_name} predict_point is not tuple/list or length is not 2')
                exit(1)

            center_x, center_y = predict_point
            center_x, center_y = round(center_x), round(center_y)
            area_id, area_mask = area_class[wav_name][0], area_class[wav_name][1]
            gt_y, gt_x = fake_location[wav_name][0], fake_location[wav_name][1]
            
            if math.sqrt((center_x - gt_x) ** 2 + (center_y - gt_y) ** 2) < 200:
                success[area_id] += 1
                
            total[area_id] += 1
            
    elif task == 5:
        for img_name, predict_point in tqdm(result.items()):
            if type(predict_point) != tuple and type(predict_point) != list or len(predict_point) != 2:
                print(f'[error] {img_name} predict_point is not tuple/list or length is not 2')
                exit(1)

            # print(img_name, predict_point)
            center_x, center_y = predict_point
            center_x, center_y = round(center_x), round(center_y)
            area_mask_all = coco.get_mask_of_img(img_name)
            expect_success[0] += (area_mask_all > 0.5).sum() / (1024 * 1280)
            if area_mask_all[center_x, center_y] > 0.5:
                success[0] += 1 
                
            total[0] += 1
    
    for i in range(3 if task in [1, 3, 4, 6] else 1):
        print(f'Area {i} success rate: {success[i]}/{total[i]} = {success[i] / total[i] * 100:.4f}%, expect success rate: {expect_success[i] / total[i] * 100 if task != 2 else math.pi * 200 * 200 / 1024 / 1280 * 100 :.4f}%')
        
    if task == 6:
        for i in range(3):
            print(f'Area {i} V-success rate: {success_v[i]}/{total[i]} = {success_v[i] / total[i] * 100:.4f}%, expect V-success rate: {expect_success_v[i] / total[i] * 100:.4f}%')

    if task == 3:
        for i in range(3):
            print(f'Area {i} A-success rate: {success_a[i]}/{total[i]} = {success_a[i] / total[i] * 100:.4f}%, expect A-success rate: {math.pi * 200 * 200 / 1024 / 1280 * 100 :.4f}%')
    
