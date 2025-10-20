import os
import json
import random
from typing import Dict, List, Tuple

import numpy as np
import cv2
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from tqdm import tqdm


TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


def get_coco_file_name(image_id_field: str) -> str:
    # Expect formats like: "COCO_val2014_000000005965.jpg" or train counterpart
    return image_id_field


def build_filename_to_imgid(coco: COCO) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for img in coco.dataset['images']:
        mapping[img['file_name']] = img['id']
    return mapping


def build_catname_to_catid(coco: COCO) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for cat in coco.loadCats(coco.getCatIds()):
        mapping[cat['name']] = cat['id']
    return mapping


def ann_for_category(anns: List[dict], target_cat_id: int) -> List[dict]:
    return [a for a in anns if a.get('category_id') == target_cat_id]


def resize_binary_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    resized = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    return (resized > 0).astype(np.uint8)


def encode_rle(mask: np.ndarray) -> dict:
    # pycocotools expects Fortran order
    rle = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
    # counts is bytes; use latin1 to losslessly serialize to JSON string
    # (later decode via .encode('latin1') before maskUtils.decode)
    rle['counts'] = rle['counts'].decode('latin1')
    return rle


def rle_area_ratio(rle: dict) -> float:
    area = float(maskUtils.area(rle))
    total = float(TARGET_WIDTH * TARGET_HEIGHT)
    return area / total if total > 0 else 0.0


def attach_masks(config_path: str, coco_json_path: str, out_path: str) -> List[dict]:
    if os.path.exists(out_path):
        return load_json(out_path)

    samples: List[dict] = load_json(config_path)
    coco = COCO(coco_json_path)

    filename_to_imgid = build_filename_to_imgid(coco)
    catname_to_catid = build_catname_to_catid(coco)

    enhanced: List[dict] = []
    missed = 0

    for idx, s in enumerate(samples):
        image_file = get_coco_file_name(s.get('image_id'))
        category_name = s.get('category')
        # Map non-COCO naming to COCO naming
        if category_name == 'plane':
            category_name = 'airplane'
        if not image_file or not category_name:
            missed += 1
            continue

        img_id = filename_to_imgid.get(image_file)
        cat_id = catname_to_catid.get(category_name)
        if img_id is None or cat_id is None:
            missed += 1
            continue

        ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=[cat_id])
        anns = coco.loadAnns(ann_ids)
        if not anns:
            missed += 1
            continue

        # If multiple anns of the category, pick the one with the largest area
        chosen = max(anns, key=lambda a: a.get('area', 0.0))

        mask = coco.annToMask(chosen)  # HxW
        resized_mask = resize_binary_mask(mask, TARGET_WIDTH, TARGET_HEIGHT)
        rle = encode_rle(resized_mask)
        ratio = rle_area_ratio(rle)

        s_new = dict(s)
        s_new['mask'] = rle
        s_new['mask_ratio'] = ratio
        enhanced.append(s_new)

    save_json(out_path, enhanced)
    print(f"Saved {len(enhanced)} samples with masks to: {out_path} (missed {missed})")
    return enhanced


def sample_by_object_size(records: List[dict], frac: float = 0.6, rng: random.Random = None) -> Dict[str, List[dict]]:
    if rng is None:
        rng = random
    groups: Dict[str, List[dict]] = {'size1': [], 'size2': [], 'size3': []}
    for r in records:
        size = r.get('object_size')
        if size in groups:
            groups[size].append(r)
    sampled: Dict[str, List[dict]] = {}
    for size, lst in groups.items():
        k = int(len(lst) * frac)
        if k <= 0:
            sampled[size] = []
        else:
            sampled[size] = rng.sample(lst, k)
    return sampled


def simulate_accuracy(sampled_groups: Dict[str, List[dict]], trials: int, rng: random.Random = None) -> Dict[str, float]:
    if rng is None:
        rng = random
    results: Dict[str, float] = {}
    for size, lst in sampled_groups.items():
        if not lst:
            results[size] = float('nan')
            continue
        # Precompute mask ratios
        ratios = [float(item.get('mask_ratio', 0.0)) for item in lst]
        # Monte Carlo via Bernoulli with p = mask_ratio
        successes = 0
        for _ in tqdm(range(trials), total=trials, desc=f"{size} sampling", leave=False):
            i = rng.randrange(len(ratios))
            p = ratios[i]
            if rng.random() < p:
                successes += 1
        results[size] = successes / trials if trials > 0 else 0.0
    return results


def mean_std(values: List[float]) -> Tuple[float, float]:
    arr = np.array(values, dtype=float)
    return float(np.nanmean(arr)), float(np.nanstd(arr, ddof=0))


def main():
    base_dir = "/home/yanhao/SSHS/AudioCOCO"
    config_path = os.path.join(base_dir, "finalConfig", "config1_depth.json")
    coco_json_path = os.path.join(base_dir, "instances_val2014.json")
    out_mask_path = os.path.join(base_dir, "finalConfig", "config1_mask.json")

    # Step 1: build masks (skip if exists)
    records_with_masks = attach_masks(config_path, coco_json_path, out_mask_path)

    # Step 2-4: Repeat 3 times: sample 60% by object_size and run 1e6 trials
    rng = random.Random()
    repeats = 3
    trials = 1_000_000

    per_size_hist: Dict[str, List[float]] = {"size1": [], "size2": [], "size3": []}

    for _ in tqdm(range(repeats), total=repeats, desc="Repeats"):
        sampled = sample_by_object_size(records_with_masks, frac=0.6, rng=rng)
        acc = simulate_accuracy(sampled, trials=trials, rng=rng)
        for size in ("size1", "size2", "size3"):
            per_size_hist[size].append(acc.get(size, float('nan')))

    # Final report
    for size in ("size1", "size2", "size3"):
        m, s = mean_std(per_size_hist[size])
        print(f"{size}: mean_accuracy={m:.6f}, std={s:.6f}")


if __name__ == "__main__":
    main()


