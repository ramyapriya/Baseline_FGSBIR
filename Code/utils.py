import torch
import cv2
import numpy as np
from dataset_utils import postprocess as post
from dataset_utils.preprocess import preprocess_data
from model.build_model import build_superpoint_model, build_maskrcnn, build_airobj
import yaml
import os

from scipy.spatial import Delaunay

def get_neighbor(vertex_id,tri):
    # get neighbor vertexes of a vertex
    helper = tri.vertex_neighbor_vertices
    index_pointers = helper[0]
    indices = helper[1]
    result_ids = indices[index_pointers[vertex_id]:index_pointers[vertex_id+1]]

    return result_ids

def get_adj(points, tri):
    adj = np.zeros((points.shape[0], points.shape[0]))

    for i in range(points.shape[0]):
        adj[i,get_neighbor(i,tri)] = 1

    return adj

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tensor_to_numpy(image):
    img = image.data.cpu().numpy()
    img = img.transpose(1, 2, 0)
    img = (img * 255.0 + 0.5).astype(np.uint8)
    img = np.clip(img, 0, 255)
    if img.shape[2] == 1:
        img = cv2.merge([img, img, img])
    else:
        img = img.copy()
    return img


def collate_self_train(batch):
    batch_mod = {'sketch_img': [], 'sketch_boxes': [],
                 'positive_img': [], 'positive_boxes': [],
                 'negative_img': [], 'negative_boxes': [],
                 }
    for i_batch in batch:
        batch_mod['sketch_img'].append(i_batch['sketch_img'])
        batch_mod['positive_img'].append(i_batch['positive_img'])
        batch_mod['negative_img'].append(i_batch['negative_img'])
        batch_mod['sketch_boxes'].append(torch.tensor(i_batch['sketch_boxes']).float())
        batch_mod['positive_boxes'].append(torch.tensor(i_batch['positive_boxes']).float())
        batch_mod['negative_boxes'].append(torch.tensor(i_batch['negative_boxes']).float())

    batch_mod['sketch_img'] = torch.stack(batch_mod['sketch_img'], dim=0)
    batch_mod['positive_img'] = torch.stack(batch_mod['positive_img'], dim=0)
    batch_mod['negative_img'] = torch.stack(batch_mod['negative_img'], dim=0)

    return batch_mod

def get_segmentation_annotation(img_path, config, save_dir=None):
    """Given an input image, get maskrcnn segmentation masks

    Args:
        img_path (str): Image path
        config (dict): MaskRCNN parameters
        save_dir (str, optional): Path to save the detections as pkl file. 
                                  Defaults to None.
        
    Returns:
        detections (list): List of detections, where each detection is a 
                           dict with ['boxes', 'labels', 'scores', 'masks'] keys
    """
    
    image_name = os.path.basename(img_path).replace('.png','')
    
    # get configs
    data_config = config['data']
    
    # model
    maskrcnn_model = build_maskrcnn(config)
    maskrcnn_model.eval()
    
    batch = {}
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.merge([image, image, image])
    image = torch.from_numpy(image).type(torch.float32)
    image = image.permute(2,0,1)
    image /= 255
    batch['image'] = [image]
    batch['image_name'] = [str(image_name)]
    
    with torch.no_grad():
        original_images = batch['image']
        original_images = [tensor_to_numpy(img.clone()) for img in original_images]

        # preprocess
        images, sizes = preprocess_data(batch, data_config)
        original_sizes = sizes['original_sizes']
        new_sizes = sizes['new_sizes']

        # model inference
        _, detections = maskrcnn_model(images, sizes) 

        # postprocess
        detections = post.postprocess_detections(new_sizes, original_sizes, detections=detections)
        
    # save results
    if save_dir is not None:
        image_names = batch['image_name']
        results = post.save_detection_results(image_names, save_dir, detections)

    return detections

def get_superpoint_points(img_path, config, save_dir=None):
    """Given an input image, get points from superpoint model

    Args:
        img_path (str): Image path
        config (dict): SuperPoint parameters
        save_dir (str, optional): Path to save the points as pkl file. Defaults to None.

    Returns:
        points (List): List of points, where each point is represented as a dict
                       with ['points', 'point_descs'] keys
    """
    image_name = os.path.basename(img_path).replace('.png','')
    
    # get configs
    data_config = config['data']
    detection_threshold = config['model']['superpoint']['detection_threshold']
    
    # model
    superpoint_model = build_superpoint_model(config)
    superpoint_model.eval()
    
    batch = {}
    src = cv2.imread(img_path)
    image = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    image = cv2.merge([image, image, image])
    image = torch.from_numpy(image).type(torch.float32)
    image = image.permute(2,0,1)
    image /= 255
    batch['image'] = [image]
    batch['image_name'] = [str(image_name)]
    
    with torch.no_grad():

        # preprocess
        images, sizes = preprocess_data(batch, data_config)
        original_sizes = sizes['original_sizes']
        new_sizes = sizes['new_sizes']

        # model inference
        points_output = superpoint_model(images) 

        # postprocess
        points = post.postprocess(new_sizes, original_sizes, detection_threshold, points_output)

    # save superpoint features 
    if save_dir is not None:
      image_names = batch['image_name']
      post.save_superpoint_features(image_names, save_dir, points_output)
    
    return points

def get_airobj_descriptor(image, seg, points, config, save_dir=None):
    """Generate AirObject Descriptor for a given image

    Args:
        image (np.ndarray): Input image
        
        seg (List): List of detections, where each detection is a 
                     dict with ['boxes', 'labels', 'scores', 'masks'] keys
                     
        points(List): List of points, where each point is represented as a dict
                       with ['points', 'point_descs'] keys
                       
        config (dict): AirObject parameters
        
        save_dir (str, optional): Path to save the points as pkl file. Defaults to None.

    Returns:
        airobj_descs (List): List of global descriptors for each object
    """    

    # model
    airobj_model = build_airobj(config)
    airobj_model.eval()
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    keypoints = points['points']
    descriptors = points['point_descs']
    
    airobj_adjs, airobj_points, airobj_descs = [], [], []
    ann_masks, ann_boxes = [], []
    for instance in range(seg['masks'].shape[0]):
        mask = seg['masks'][instance, 0, :, :].numpy()
        object_filter = mask[keypoints[:,0].T,keypoints[:,1].T]
        np_obj_pts = keypoints[np.where(object_filter==1)[0]].numpy()
        
        try:
            tri = Delaunay(np_obj_pts, qhull_options='QJ')
        except:
            continue
        
        adj = get_adj(np_obj_pts, tri)

        airobj_adjs.append(torch.from_numpy(adj).float().to(device))
        airobj_points.append(keypoints[np.where(object_filter==1)[0]].float().to(device))
        airobj_descs.append(descriptors[np.where(object_filter==1)[0]].float().to(device))
        
        inds = np.array(np.where(mask))
        y1, x1 = np.amin(inds, axis=1)
        y2, x2 = np.amax(inds, axis=1)

        ann_masks.append(mask)
        ann_boxes.append(np.array([x1, y1, x2, y2]))
        
    if len(ann_masks) == 0:
        return None
    
    with torch.no_grad():
        airobj_obj_descs = []
        for k in range(len(airobj_points)):
            airobj_obj_descs.append(airobj_model([airobj_points[k]], [airobj_descs[k]], [airobj_adjs[k]]))
        airobj_obj_descs = torch.cat(airobj_obj_descs)
        
    if save_dir:
        # results = viz.save_detection_results([image], ['output.png'], save_dir,
        #                                     [seg], None, [points], True, False)
        pass
    
    return airobj_descs
    
    
def collate_self_test(batch):
    batch_mod = {'sketch_img': [], 'sketch_boxes': [], 'sketch_path': [],
                 'positive_img': [], 'positive_boxes': [], 'positive_path': [],
                 }

    for i_batch in batch:
        batch_mod['sketch_img'].append(i_batch['sketch_img'])
        batch_mod['sketch_path'].append(i_batch['sketch_path'])
        batch_mod['positive_img'].append(i_batch['positive_img'])
        batch_mod['positive_path'].append(i_batch['positive_path'])
        batch_mod['sketch_boxes'].append(torch.tensor(i_batch['sketch_boxes']).float())
        batch_mod['positive_boxes'].append(torch.tensor(i_batch['positive_boxes']).float())

    batch_mod['sketch_img'] = torch.stack(batch_mod['sketch_img'], dim=0)
    batch_mod['positive_img'] = torch.stack(batch_mod['positive_img'], dim=0)

    return batch_mod

if __name__ == "__main__":
    img_path = '/Users/ramya/University/misc/Baseline_FGSBIR/Dataset/ShoeV2/photo/1031000079.png'
    save_dir = '/Users/ramya/University/misc/Baseline_FGSBIR/Dataset/ShoeV2/seg_results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    image = cv2.imread(img_path)    
    
    maskrcnn_config_file = '/Users/ramya/University/misc/Baseline_FGSBIR/Code/config/maskrcnn_inference.yaml'
    with open(maskrcnn_config_file, 'r', encoding='utf-8') as f:
        config = f.read()
        config = yaml.safe_load(config)
        seg = get_segmentation_annotation(img_path, config, save_dir=None)
    f.close()
    
    superpoint_config_file = '/Users/ramya/University/misc/Baseline_FGSBIR/Code/config/superpoint_inference.yaml'
    with open(superpoint_config_file, 'r', encoding='utf-8') as f:
        config = f.read()
        config = yaml.safe_load(config)
        points = get_superpoint_points(img_path, config, save_dir=None)
    f.close()    
    
    airobj_config_file = '/Users/ramya/University/misc/Baseline_FGSBIR/Code/config/airobj_inference.yaml'
    with open(airobj_config_file, 'r', encoding='utf-8') as f:
        config = f.read()
        config = yaml.safe_load(config)
        import pdb; pdb.set_trace()
        airobj_desc = get_airobj_descriptor(image, seg, points, config, save_dir=None)
    f.close()