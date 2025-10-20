import json
import io
import requests
from pycocotools.coco import COCO
from PIL import Image

trainCOCO = COCO('./annotations/instances_train2014.json')
valCOCO = COCO('./annotations/instances_val2014.json')

with open('/home/yanhao/SSHS/AudioCOCO/finalConfig/config2.json', 'r') as file:
    config2_data = json.load(file)

with open('/home/yanhao/SSHS/AudioCOCO/finalConfig/config3.json', 'r') as file:
    config3_data = json.load(file)

for data in config2_data:
    imageID = int(str(data['image_id'].split('.')[0].split("_")[-1]))
    imageData = trainCOCO.loadImgs(imageID)[0]

    image_url = imageData['coco_url']
    img_data = requests.get(image_url).content

    image = Image.open(io.BytesIO(img_data))
    imgResize = image.resize((1920, 1080))
    savePath = './images/' + data['image_id']
    imgResize.save(savePath)


for data in config3_data:
    imageID = int(str(data['image_id'].split('.')[0].split("_")[-1]))
    imageData = valCOCO.loadImgs(imageID)[0]

    image_url = imageData['coco_url']
    img_data = requests.get(image_url).content

    image = Image.open(io.BytesIO(img_data))
    imgResize = image.resize((1920, 1080))
    savePath = './images/' + data['image_id']
    imgResize.save(savePath)