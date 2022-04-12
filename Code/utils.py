import torch
import torch.nn.functional as F
import cv2
import numpy as np
from dataset_utils import postprocess as post
from dataset_utils.preprocess import preprocess_data
from model.build_model import build_superpoint_model, build_maskrcnn, build_airobj
import yaml
import os, time
from scipy.spatial import Delaunay
import viz

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

def get_segmentation_annotation(image, maskrcnn_model, config, save_dir=None):
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
    
    # image_name = os.path.basename(img_path).replace('.png','')
    
    # get configs
    data_config = config['data']
    batch = {}

    if len(image.shape) < 2 or len(image.shape) > 3:
        assert('Invalid type: Image ')
        return
    
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    image = cv2.merge([image, image, image])
    image = torch.from_numpy(image).type(torch.float32)
    image = image.permute(2,0,1)
    image /= 255
    batch['image'] = [image]
    batch['image_name'] = ['output_mask.png']

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
        detections = post.postprocess_detections(new_sizes, original_sizes, conf_threshold=0.3, detections=detections)
        
    # save results
    if save_dir is not None:
        image_names = batch['image_name']
        results = post.save_detection_results(image_names, save_dir, detections)

    return detections

def get_superpoint_points(image, superpoint_model, config, save_dir=None):
    """Given an input image, get points from superpoint model

    Args:
        img_path (str): Image path
        config (dict): SuperPoint parameters
        save_dir (str, optional): Path to save the points as pkl file. Defaults to None.

    Returns:
        points (List): List of points, where each point is represented as a dict
                       with ['points', 'point_descs'] keys
    """
    # image_name = os.path.basename(img_path).replace('.png','')
    
    # get configs
    data_config = config['data']
    detection_threshold = config['model']['superpoint']['detection_threshold']
    
    batch = {}

    if len(image.shape) < 2 or len(image.shape) > 3:
        assert('Invalid type: Image ')
        return
    
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    image = cv2.merge([image, image, image])
    image = torch.from_numpy(image).type(torch.float32)
    image = image.permute(2,0,1)
    image /= 255
    batch['image'] = [image]
    # batch['image_name'] = [str(image_name)]
    batch['image_name'] = 'output_points.png'
    
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
    
    points = points[0]
    seg = seg[0]
    keypoints = points['points']
    descriptors = points['point_descs']
    
    airobj_adjs, airobj_points, airobj_descs = [], [], []
    ann_masks, ann_boxes = [], []
    tri_objects = []
    for instance in range(seg['masks'].shape[0]):
        mask = seg['masks'][instance, 0, :, :].numpy()
        object_filter = mask[keypoints[:,0].T,keypoints[:,1].T]
        np_obj_pts = keypoints[np.where(object_filter==1)[0]].numpy()
        
        try:
            tri = Delaunay(np_obj_pts, qhull_options='QJ')
        except:
            continue
        
        adj = get_adj(np_obj_pts, tri)
        tri_objects.append(tri)
        airobj_adjs.append(torch.from_numpy(adj).float().to(device)) # Size: NxN
        airobj_points.append(keypoints[np.where(object_filter==1)[0]].float().to(device)) # Size: Nx2
        airobj_descs.append(descriptors[np.where(object_filter==1)[0]].float().to(device)) # Size: NxD
        
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
            airobj_obj_descs.append(airobj_model([airobj_points[k]], [airobj_descs[k]], [airobj_adjs[k]])) # Size: 1x2048
        airobj_obj_descs = torch.cat(airobj_obj_descs)
        
    if save_dir:
        results = viz.save_detection_results([image], ['output.png'], save_dir, [seg], None, [points], [tri_objects], True, True)
    
    return airobj_obj_descs

def collate_self_train(batch):
    batch_mod = {'sketch_img': [], 'sketch_img_mask': [], 'sketch_img_points': [], 'sketch_img_descs': [],
                 'positive_img': [], 'positive_img_mask': [], 'positive_img_points': [], 'positive_img_descs': [],
                 'negative_img': [], 'negative_img_mask': [], 'negative_img_points': [], 'negative_img_descs': []
                }
    for i_batch in batch:
        batch_mod['sketch_img'].append(i_batch['sketch_img'])
        batch_mod['positive_img'].append(i_batch['positive_img'])
        batch_mod['negative_img'].append(i_batch['negative_img'])
        batch_mod['sketch_img_mask'].append(i_batch['sketch_img_mask'][0]['masks'])
        batch_mod['sketch_img_points'].append(i_batch['sketch_img_points'][0]['points'])
        batch_mod['sketch_img_descs'].append(i_batch['sketch_img_points'][0]['point_descs'])
        batch_mod['positive_img_mask'].append(i_batch['positive_img_mask'][0]['masks'])
        batch_mod['positive_img_points'].append(i_batch['positive_img_points'][0]['points'])
        batch_mod['positive_img_descs'].append(i_batch['positive_img_points'][0]['point_descs'])
        batch_mod['negative_img_mask'].append(i_batch['negative_img_mask'][0]['masks'])
        batch_mod['negative_img_points'].append(i_batch['negative_img_points'][0]['points'])
        batch_mod['negative_img_descs'].append(i_batch['negative_img_points'][0]['point_descs'])

    batch_mod['sketch_img'] = torch.stack(batch_mod['sketch_img'], dim=0)
    batch_mod['positive_img'] = torch.stack(batch_mod['positive_img'], dim=0)
    batch_mod['negative_img'] = torch.stack(batch_mod['negative_img'], dim=0)

    return batch_mod
    
    
def collate_self_test(batch):
    batch_mod = {'sketch_img': [], 'sketch_img_mask': [], 'sketch_img_points': [], 'sketch_img_descs': [],
                 'positive_img': [], 'positive_img_mask': [], 'positive_img_points': [], 'positive_img_descs': [],
                 'sketch_path': [], 'positive_path': [] 
                }
    for i_batch in batch:
        batch_mod['sketch_img'].append(i_batch['sketch_img'])
        batch_mod['sketch_path'].append(i_batch['sketch_path'])
        batch_mod['positive_img'].append(i_batch['positive_img'])
        batch_mod['positive_path'].append(i_batch['positive_path'])
        batch_mod['sketch_img_mask'].append(i_batch['sketch_img_mask'][0]['masks'])
        batch_mod['sketch_img_points'].append(i_batch['sketch_img_points'][0]['points'])
        batch_mod['sketch_img_descs'].append(i_batch['sketch_img_points'][0]['point_descs'])
        batch_mod['positive_img_mask'].append(i_batch['positive_img_mask'][0]['masks'])
        batch_mod['positive_img_points'].append(i_batch['positive_img_points'][0]['points'])
        batch_mod['positive_img_descs'].append(i_batch['positive_img_points'][0]['point_descs'])

    batch_mod['sketch_img'] = torch.stack(batch_mod['sketch_img'], dim=0)
    batch_mod['positive_img'] = torch.stack(batch_mod['positive_img'], dim=0)

    return batch_mod

def get_batch_objects(batch, device, input_type):
    """
    batch_size 

    """
    batch_size = batch['sketch_img'].shape[0]
    batch_points, batch_descs, batch_adjs, iids = [], [], [], []
    for idx in range(batch_size):
        
        mask = batch[input_type+'_img_mask'][idx]
        keypoints =  batch[input_type+'_img_points'][idx]
        descriptors =  batch[input_type+'_img_descs'][idx]
        object_filter = torch.ones(keypoints.shape[0])

        if torch.numel(mask): # If segmentation fails, include all kepypoints for 
                              # Delaunay Triangulation
            mask = mask[0].squeeze()
            object_filter = mask[keypoints[:,0].T,keypoints[:,1].T]
            np_obj_pts = keypoints[torch.where(object_filter==1.0)[0]].numpy()
        else:
            np_obj_pts = keypoints.numpy()

        if not torch.numel(keypoints) or not np.size(np_obj_pts) or np_obj_pts.shape[0] < 2:
            iids.append(idx)
        try:
            tri = Delaunay(np_obj_pts, qhull_options='QJ')
            adj = get_adj(np_obj_pts, tri)
        except:
            adj = np.ones((np_obj_pts.shape[0], np_obj_pts.shape[0]))
            # iids.append(idx)

        batch_adjs.append(torch.from_numpy(adj).float().to(device))
        batch_points.append(keypoints[np.where(object_filter==1)[0]].float().to(device))
        batch_descs.append(descriptors[np.where(object_filter==1)[0]].float().to(device))
    
    return batch_points, batch_descs, batch_adjs, iids

def evaluate(dataloader_test, model):
    Image_Feature_ALL = []
    Image_Name = []
    Sketch_Feature_ALL = []
    Sketch_Name = []
    start_time = time.time()
    model.eval()
    
    for idx, sampled_batch in enumerate(dataloader_test):
        
        anchor_points, anchor_descs, anchor_adjs, iids = get_batch_objects(sampled_batch, device, 'sketch')
        positive_points, positive_descs, positive_adjs, iids = get_batch_objects(sampled_batch, device, 'positive')
        

        if len(anchor_points) == 1 and not torch.numel(anchor_points[0]):
            return -1, -1
        sketch_feature = model(anchor_points, anchor_descs, anchor_adjs)
        positive_feature = model(positive_points, positive_descs, positive_adjs)

        Sketch_Feature_ALL.extend(sketch_feature)
        Sketch_Name.extend(sampled_batch['sketch_path'])

        for i_num, positive_name in enumerate(sampled_batch['positive_path']):
            if positive_name not in Image_Name:
                Image_Name.append(sampled_batch['positive_path'][i_num])
                Image_Feature_ALL.append(positive_feature[i_num])

    rank = torch.zeros(len(Sketch_Name))
    Image_Feature_ALL = torch.stack(Image_Feature_ALL)

    for num, sketch_feature in enumerate(Sketch_Feature_ALL):
        s_name = Sketch_Name[num]
        sketch_query_name = '_'.join(s_name.split('/')[-1].split('_')[:-1])
        position_query = Image_Name.index(sketch_query_name)

        distance = F.pairwise_distance(sketch_feature.unsqueeze(0), Image_Feature_ALL)
        target_distance = F.pairwise_distance(sketch_feature.unsqueeze(0),
                                                Image_Feature_ALL[position_query].unsqueeze(0))

        rank[num] = distance.le(target_distance).sum()

    top1 = rank.le(1).sum().numpy() / rank.shape[0]
    top10 = rank.le(10).sum().numpy() / rank.shape[0]

    print('Time to EValuate:{}'.format(time.time() - start_time))
    return top1, top10