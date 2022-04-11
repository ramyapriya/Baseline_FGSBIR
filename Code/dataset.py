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
from utils import get_segmentation_annotation, get_superpoint_points
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FGSBIR_Dataset(data.Dataset):
    def __init__(self, hp, mode):

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
        
        self.segmentation_

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
            sketch_img = rasterize_Sketch(vector_x)
            sketch_img = Image.fromarray(sketch_img).convert('RGB')

            positive_img = Image.open(positive_path).convert('RGB')
            negative_img = Image.open(negative_path).convert('RGB')

            n_flip = random.random()
            if n_flip > 0.5:
                sketch_img = F.hflip(sketch_img)
                positive_img = F.hflip(positive_img)
                negative_img = F.hflip(negative_img)

            # For all 3 images, get segmentation masks
            sketch_img_mask = get_segmentation_annotation(sketch_img, self.hp['seg_configs'])
            positive_img_mask = get_segmentation_annotation(positive_img, self.hp['seg_configs'])
            negative_img_mask = get_segmentation_annotation(negative_img, self.hp['seg_configs'])
            
            # For all 3 images, get superpoints
            sketch_img_points = get_superpoint_points(sketch_img, self.hp['superpoint_configs'])
            positive_img_points = get_superpoint_points(positive_img, self.hp['superpoint_configs'])
            negative_img_points = get_superpoint_points(negative_img, self.hp['superpoint_configs'])
            
            # Transforms
            sketch_img = self.train_transform(sketch_img)
            positive_img = self.train_transform(positive_img)
            negative_img = self.train_transform(negative_img)

            sample = {'sketch_img': sketch_img, 'sketch_path': sketch_path,
                      'sketch_mask': sketch_img_mask, 'sketch_points': sketch_img_points,
                      'positive_img': positive_img, 'positive_path': positive_sample,
                      'positive_mask': positive_img_mask, 'positive_points': positive_img_points,
                      'negative_img': negative_img, 'negative_path': negative_sample,
                      'negative_mask': negative_img_mask, 'negative_points': negative_img_points
                      }

        elif self.mode == 'Test':

            sketch_path = self.Test_Sketch[item]
            vector_x = self.Coordinate[sketch_path]
            sketch_img = rasterize_Sketch(vector_x)
            sketch_img_PIL = Image.fromarray(sketch_img).convert('RGB')
            

            positive_sample = '_'.join(self.Test_Sketch[item].split('/')[-1].split('_')[:-1])
            positive_path = os.path.join(self.root_dir, 'photo', positive_sample + '.png')
            positive_img_PIL = Image.open(positive_path).convert('RGB')
            positive_img_np = np.asarray(positive_img_PIL, dtype=np.uint8)
            
            # For all 3 images, get segmentation masks
            sketch_img_mask = get_segmentation_annotation(sketch_img, self.hp['seg_configs'])
            positive_img_mask = get_segmentation_annotation(positive_img_np, self.hp['seg_configs'])
            
            # For all 3 images, get superpoints
            sketch_img_points = get_superpoint_points(sketch_img, self.hp['superpoint_configs'])
            positive_img_points = get_superpoint_points(positive_img_np, self.hp['superpoint_configs'])
            
            # Transforms
            sketch_img = self.test_transform(sketch_img_PIL)
            positive_img = self.test_transform(positive_img_PIL)
            
            sample = {'sketch_img': sketch_img, 'sketch_path': sketch_path, 'Coordinate':vector_x,
                      'positive_img': positive_img, 'positive_path': positive_sample,
                      'sketch_mask': sketch_img_mask, 'sketch_points': sketch_img_points,
                      'positive_mask': positive_img_mask, 'positive_points': positive_img_points
                      }

        return sample

    def __len__(self):
        if self.mode == 'Train':
            return len(self.Train_Sketch)
        elif self.mode == 'Test':
            return len(self.Test_Sketch)

def get_dataloader(hp, dataset_type='FGSBIR'):
    
    if dataset_type == 'FGSBIR':
        dataset_Train  = FGSBIR_Dataset(hp, mode = 'Train')
        dataset_Test  = FGSBIR_Dataset(hp, mode = 'Test')
    elif dataset_type == 'FGSBIR_AirObj':
        dataset_Train  = FGSBIR_Dataset(hp, mode = 'Train')
        dataset_Test  = FGSBIR_Dataset(hp, mode = 'Test')
    
    dataloader_Train = data.DataLoader(dataset_Train, batch_size=hp.batchsize, shuffle=True,
                                         num_workers=int(hp.nThreads))

    dataloader_Test = data.DataLoader(dataset_Test, batch_size=1, shuffle=False,
                                         num_workers=int(hp.nThreads))

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
