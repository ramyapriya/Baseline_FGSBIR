import torch
import pickle
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from random import randint
from PIL import Image
import random
import torchvision.transforms.functional as F
from rasterize import rasterize_Sketch
from utils import get_segmentation_annotation, get_superpoint_points, collate_self_train, collate_self_test
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from model.build_model import build_superpoint_model, build_maskrcnn

class FGSBIR_Dataset(data.Dataset):
    def __init__(self, hp, configs, mode):

        self.hp = hp
        self.mode = mode
        coordinate_path = os.path.join(hp.root_dir, 'Dataset', hp.dataset_name , hp.dataset_name + '_Coordinate')
        self.root_dir = os.path.join(hp.root_dir, 'Dataset', hp.dataset_name)
        with open(coordinate_path, 'rb') as fp:
            self.Coordinate = pickle.load(fp)

        self.Train_Sketch = [x for x in self.Coordinate if 'train' in x]
        self.Test_Sketch = [x for x in self.Coordinate if 'test' in x]

        self.train_transform = get_ransform('Train')
        self.test_transform = get_ransform('Test')

        self.configs = configs

        # maskrcnn model
        self.maskrcnn_model = build_maskrcnn(configs['seg_configs'])
        self.maskrcnn_model.eval()
        print('Loaded MaskRCNN model')

        # superpoint_model
        self.superpoint_model = build_superpoint_model(configs['superpoint_configs'])
        self.superpoint_model.eval()
        print('Loaded Superpoint extraction model')


    def __getitem__(self, item):
        sample  = {}
        if self.mode == 'Train':
            
            sketch_path = self.Train_Sketch[item]

            positive_sample = '_'.join(self.Train_Sketch[item].split('/')[-1].split('_')[:-1])
            positive_path = os.path.join(self.root_dir, 'photo', positive_sample + '.png')

            possible_list = list(range(len(self.Train_Sketch)))
            possible_list.remove(item)
            negative_item = possible_list[randint(0, len(possible_list) - 1)]
            negative_sample = '_'.join(self.Train_Sketch[negative_item].split('/')[-1].split('_')[:-1])
            negative_path = os.path.join(self.root_dir, 'photo', negative_sample + '.png')

            vector_x = self.Coordinate[sketch_path]
            sketch_img_rasterized = rasterize_Sketch(vector_x)
            sketch_img = Image.fromarray(sketch_img_rasterized).convert('RGB')

            positive_img = Image.open(positive_path).convert('RGB')
            negative_img = Image.open(negative_path).convert('RGB')

            n_flip = random.random()
            if n_flip > 0.5:
                sketch_img = F.hflip(sketch_img)
                positive_img = F.hflip(positive_img)
                negative_img = F.hflip(negative_img)


            # Convert to numpy
            positive_img_np = np.asarray(positive_img, dtype=np.uint8)
            negative_img_np = np.asarray(negative_img, dtype=np.uint8)
            # import pdb; pdb.set_trace()

            # For all 3 images, get segmentation masks
            sketch_img_mask = get_segmentation_annotation(sketch_img_rasterized, self.maskrcnn_model, self.configs['seg_configs'])
            positive_img_mask = get_segmentation_annotation(positive_img_np, self.maskrcnn_model, self.configs['seg_configs'])
            negative_img_mask = get_segmentation_annotation(negative_img_np, self.maskrcnn_model, self.configs['seg_configs'])
            
            # For all 3 images, get superpoints
            sketch_img_points = get_superpoint_points(sketch_img_rasterized, self.superpoint_model, self.configs['superpoint_configs'])
            positive_img_points = get_superpoint_points(positive_img_np, self.superpoint_model, self.configs['superpoint_configs'])
            negative_img_points = get_superpoint_points(negative_img_np, self.superpoint_model, self.configs['superpoint_configs'])
            
            # Transforms
            sketch_img = self.train_transform(sketch_img)
            positive_img = self.train_transform(positive_img)
            negative_img = self.train_transform(negative_img)

            sample = {'sketch_img': sketch_img, 'sketch_path': sketch_path,
                      'sketch_img_mask': sketch_img_mask, 'sketch_img_points': sketch_img_points,
                      'positive_img': positive_img, 'positive_path': positive_sample,
                      'positive_img_mask': positive_img_mask, 'positive_img_points': positive_img_points,
                      'negative_img': negative_img, 'negative_path': negative_sample,
                      'negative_img_mask': negative_img_mask, 'negative_img_points': negative_img_points
                      }

        elif self.mode == 'Test':

            sketch_path = self.Test_Sketch[item]
            vector_x = self.Coordinate[sketch_path]
            sketch_img_rasterized = rasterize_Sketch(vector_x)
            sketch_img_PIL = Image.fromarray(sketch_img_rasterized).convert('RGB')
            

            positive_sample = '_'.join(self.Test_Sketch[item].split('/')[-1].split('_')[:-1])
            positive_path = os.path.join(self.root_dir, 'photo', positive_sample + '.png')
            positive_img_PIL = Image.open(positive_path).convert('RGB')
            positive_img_np = np.asarray(positive_img_PIL, dtype=np.uint8)
            
            # For both images, get segmentation masks
            sketch_img_mask = get_segmentation_annotation(sketch_img_rasterized, self.maskrcnn_model, self.configs['seg_configs'])
            positive_img_mask = get_segmentation_annotation(positive_img_np, self.maskrcnn_model, self.configs['seg_configs'])
            
            # For both images, get superpoints
            sketch_img_points = get_superpoint_points(sketch_img_rasterized, self.superpoint_model, self.configs['superpoint_configs'])
            positive_img_points = get_superpoint_points(positive_img_np, self.superpoint_model, self.configs['superpoint_configs'])
            
            # Transforms
            sketch_img = self.test_transform(sketch_img_PIL)
            positive_img = self.test_transform(positive_img_PIL)
            
            sample = {'sketch_img': sketch_img, 'sketch_path': sketch_path, 'Coordinate':vector_x,
                      'positive_img': positive_img, 'positive_path': positive_sample,
                      'sketch_img_mask': sketch_img_mask, 'sketch_img_points': sketch_img_points,
                      'positive_img_mask': positive_img_mask, 'positive_img_points': positive_img_points
                      }

        return sample

    def __len__(self):
        if self.mode == 'Train':
            return len(self.Train_Sketch)
        elif self.mode == 'Test':
            return len(self.Test_Sketch)

def get_dataloader(hp, configs=None, dataset_type='FGSBIR'):
    
    if dataset_type == 'FGSBIR':
        dataset_Train  = FGSBIR_Dataset(hp, mode = 'Train')
        dataset_Test  = FGSBIR_Dataset(hp, mode = 'Test')
    elif dataset_type == 'FGSBIR_AirObj':
        dataset_Train  = FGSBIR_Dataset(hp, configs, mode = 'Train')
        dataset_Test  = FGSBIR_Dataset(hp, configs, mode = 'Test')
    
    dataloader_Train = data.DataLoader(dataset_Train, batch_size=hp.batchsize, collate_fn=collate_self_train, shuffle=True)


    dataloader_Test = data.DataLoader(dataset_Test, batch_size=1, collate_fn=collate_self_test, shuffle=False)

    # debug
    # batch = next(iter(dataloader_Train))
    
    return dataloader_Train, dataloader_Test

def get_ransform(type):
    transform_list = []
    if type is 'Train':
        transform_list.extend([transforms.Resize(299)])
    elif type is 'Test':
        transform_list.extend([transforms.Resize(299)])
    transform_list.extend(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transforms.Compose(transform_list)
