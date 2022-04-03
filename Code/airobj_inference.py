import yaml
import os
import cv2
from utils import get_segmentation_annotation, get_superpoint_points, get_airobj_descriptor

if __name__ == "__main__":
    img_path = './Dataset/ShoeV2/photo/1031000079.png'
    save_dir = './Dataset/ShoeV2/results'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    image = cv2.imread(img_path)
    
    maskrcnn_config_file = './Code/config/maskrcnn_inference.yaml'
    with open(maskrcnn_config_file, 'r', encoding='utf-8') as f:
        config = f.read()
        config = yaml.safe_load(config)
    seg = get_segmentation_annotation(img_path, config, save_dir=None)
    f.close()
    
    superpoint_config_file = './Code/config/superpoint_inference.yaml'
    with open(superpoint_config_file, 'r', encoding='utf-8') as f:
        config = f.read()
        config = yaml.safe_load(config)
    points = get_superpoint_points(img_path, config, save_dir=None)
    f.close()    
    
    airobj_config_file = './Code/config/airobj_inference.yaml'
    with open(airobj_config_file, 'r', encoding='utf-8') as f:
        config = f.read()
        config = yaml.safe_load(config)
    airobj_desc = get_airobj_descriptor(image, seg, points, config, save_dir=None)
    f.close()