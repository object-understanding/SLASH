import os
import math
import random
import json
import numpy as np
from PIL import Image
import torch
import h5py
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import get_worker_info, IterableDataset
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F 
from pycocotools import mask
from collections import defaultdict

from utils.common import get_rank, get_world_size

class CLEVR(Dataset):
    def __init__(self, data_dir, phase='train', sub=False, cfg=None):
        super(CLEVR, self).__init__()
        assert phase in ['train', 'val', 'test']
        assert cfg.WEAK_SUP.TYPE in ['', 'bbox', 'bbox_center']
        assert not sub or (sub and cfg.WEAK_SUP.SPLIT.RATIO < 1)

        self.sub = sub
        self.phase = phase
        self.img_size = cfg.DATA.IMG_SIZE
        self.num_slots = cfg.MODEL.SLOT.NUM
        self.image_dir = os.path.join(data_dir, 'images', self.phase)
        self.scene_dir = os.path.join(data_dir, 'scenes')
        self.metadata = json.load(open(os.path.join(self.scene_dir, f"CLEVR_{self.phase}_scenes.json")))
        
        self.files = sorted(os.listdir(self.image_dir))
        self.len_files = len(self.files)

        self.crop_size = cfg.DATA.CROP_SIZE
        if self.crop_size > 0:
            self.img_train_transform = transforms.Compose([
                # TODO: center crop? [29: 221, 64: 256] in official google github
                transforms.CenterCrop(self.crop_size),
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor()])

            self.img_val_transform = transforms.Compose([
                transforms.CenterCrop(self.crop_size),
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor()])
        else:
            self.img_train_transform = transforms.Compose([
                # TODO: center crop? [29: 221, 64: 256] in official google github
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor()])

            self.img_val_transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor()])
    
        self.center_crop = transforms.CenterCrop(self.crop_size)
        self.mask_resize = transforms.Resize((self.img_size, self.img_size))


    def __getitem__(self, index):
        index = index % self.len_files
        
        filename = self.metadata['scenes'][index]['image_filename']
        while filename not in self.files:
            index = random.randint(0, len(self.files))
            filename = self.metadata['scenes'][index]['image_filename']

        image = Image.open(os.path.join(self.image_dir, filename)).convert("RGB")
        w, h = image.size

        image = self.img_train_transform(image)
        sample = {'image': image}

        '''
        coords = target[:2]
        object_size = torch.argmax(target[2:4])
        material = torch.argmax(target[4:6])
        shape = torch.argmax(target[6:9])
        color = torch.argmax(target[9:17])
        real_obj = target[17]
        '''
        pixel_coords = torch.full((self.num_slots, 2), fill_value=-1, dtype=torch.float32)
        object_size = torch.zeros((self.num_slots, 2), dtype=torch.long)
        material = torch.zeros((self.num_slots, 2), dtype=torch.long)
        shape = torch.zeros((self.num_slots, 3), dtype=torch.long)
        color = torch.zeros((self.num_slots, 8), dtype=torch.long)
        real_obj = torch.zeros((self.num_slots, 1), dtype=torch.long)
        
        objects = self.metadata['scenes'][index]['objects']

        # TODO: currently, only real objects is considered. 
        for obj_i in range(1, len(objects)):
            obj = objects[obj_i]
            pixel_coords[obj_i][0] = obj['pixel_coords'][0] / w
            pixel_coords[obj_i][1] = obj['pixel_coords'][1] / h
            object_size[obj_i][obj['size']-1] = 1
            material[obj_i][obj['material']-1] = 1
            shape[obj_i][obj['shape']-1] = 1
            color[obj_i][obj['color']-1] = 1
            real_obj[obj_i][0] = 1
        
        sample['pixel_coords'] = pixel_coords - 0.5
        sample['object_size'] = object_size
        sample['material'] = material
        sample['shape'] = shape
        sample['color'] = color
        sample['real_obj'] = real_obj

        return sample
        
    def __len__(self):
        # make sub dataset longer
        # to prevent early stop in dataloader when zip original and sub loader
        return self.len_files


class PTR(Dataset):
    def __init__(self, data_dir, phase='train', sub=False, cfg=None):
        super(PTR, self).__init__()
        
        assert phase in ['train', 'val', 'test']

        self.sub = sub
        self.phase = phase 
        self.img_size = cfg.DATA.IMG_SIZE
        self.num_slots = cfg.MODEL.SLOT.NUM
        self.image_dir = os.path.join(data_dir, 'images', self.phase)
        self.scene_dir = os.path.join(data_dir, 'scenes', self.phase)

        self.files = sorted(os.listdir(self.image_dir))
        self.len_files = len(self.files)
        
        self.img_train_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()])
        
        self.img_val_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()])
            
        self.mask_resize = transforms.Resize((self.img_size, self.img_size))

        self.category_to_idx = {"Chair":0, "Bed":1, "Table":2, "Refrigerator":3, "Cart":4}

    def __getitem__(self, index):
        index = index % self.len_files

        image_name = self.files[index]
        image = Image.open(os.path.join(self.image_dir, image_name)).convert("RGB")    
        w, h = image.size

        scene_name = image_name[:-3] + 'json'
        metadata = json.load(open(os.path.join(self.scene_dir, scene_name)))

        image = self.img_train_transform(image)
        sample = {'image': image}

        '''
        coords = (x, y)
        category = # ["Chair", "Bed", "Table", "Refrigerator", "Cart"]
        real_obj = 0/1
        '''
        pixel_coords = torch.full((self.num_slots, 2), fill_value=-1, dtype=torch.float32)
        category = torch.zeros((self.num_slots, 5), dtype=torch.long)
        real_obj = torch.zeros((self.num_slots, 1), dtype=torch.long)

        for obj_i in range(len(metadata['objects'])):
            bbox = metadata['objects'][obj_i]['bbox']
            pixel_coords[obj_i] = torch.tensor([ (bbox[0] + bbox[1]) / 2 / w, (bbox[2] + bbox[3]) / 2/ h ])
            category[obj_i][self.category_to_idx[metadata['objects'][obj_i]['category']]] = 1
            real_obj[obj_i][0] = 1

        sample['pixel_coords'] = pixel_coords - 0.5
        sample['category'] = category
        sample['real_obj'] = real_obj

        return sample
        
    def __len__(self):
        return self.len_files
        

class MOVi(Dataset):
    def __init__(self, data_dir, phase='train', sub=False, cfg=None):
        super(MOVi, self).__init__()
        assert phase in ['train', 'val', 'val_from_train']

        self.sub = sub
        self.phase = phase
        self.img_size = cfg.DATA.IMG_SIZE
        self.num_slots = cfg.MODEL.SLOT.NUM
        self.image_dir = os.path.join(data_dir, 'images', self.phase)
        self.scene_dir = os.path.join(data_dir, 'scenes')
        self.files = sorted(os.listdir(self.image_dir))
        self.len_files = len(self.files)

        self.crop_size = cfg.DATA.CROP_SIZE 
        if self.crop_size > 0:
            self.img_train_transform = transforms.Compose([
                transforms.CenterCrop(self.crop_size),
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor()])

            self.img_val_transform = transforms.Compose([
                transforms.CenterCrop(self.crop_size),
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor()])
        else:
            self.img_train_transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor()])

            self.img_val_transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor()])
    
        self.center_crop = transforms.CenterCrop(self.crop_size)
        self.mask_resize = transforms.Resize((self.img_size, self.img_size))

        self.categories = ["Action Figures", "Bag", "Board Games", 
                           "Bottles and Cans and Cups", "Camera", 
                           "Car Seat", "Consumer Goods", "Hat", 
                           "Headphones", "Keyboard", "Legos", 
                           "Media Cases", "Mouse", "None", "Shoe", 
                           "Stuffed Toys", "Toys"]
        self.category_to_idx =  dict()
        for i, category in enumerate(self.categories):
            self.category_to_idx[category] = i

    def __getitem__(self, index):
        index = index % self.len_files

        image_name = self.files[index]
        image = Image.open(os.path.join(self.image_dir, image_name)).convert("RGB") 
        w, h = image.size

        scene_name = image_name[:-3] + 'json'
        metadata = json.load(open(os.path.join(self.scene_dir, self.phase, scene_name)))

        image = self.img_train_transform(image)
        sample = {'image': image}


        '''
        coords = (x, y)
        category = self.categories
        real_obj = 0/1
        '''
        pixel_coords = torch.full((self.num_slots, 2), fill_value=-1, dtype=torch.float32)
        category = torch.zeros((self.num_slots, len(self.categories)), dtype=torch.long)
        real_obj = torch.zeros((self.num_slots, 1), dtype=torch.long)

        for obj_i in range(len(metadata['instances'])):
            '''
            bboxes: (None, 4) [float32]
                The normalized image-space (2D) coordinates of the bounding box
                [ymin, xmin, ymax, xmax] for all the frames in which the object is visible
                (as specified in bbox_frames).
            ''' 
            bbox = metadata['instances'][obj_i]['bboxes'][0]
            pixel_coords[obj_i] = torch.tensor([ (bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2 ])
            category[obj_i][self.category_to_idx[metadata['instances'][obj_i]['category']]] = 1
            real_obj[obj_i][0] = 1

        sample['pixel_coords'] = pixel_coords - 0.5
        sample['category'] = category
        sample['real_obj'] = real_obj

        return sample
        
    def __len__(self):
        return self.len_files


class CLEVRTEX(Dataset):
    def __init__(self, data_dir, phase='train', sub=False, cfg=None):
        super(CLEVRTEX, self).__init__()
        assert phase in ['train', 'val']

        self.sub = sub
        self.phase = phase
        self.img_size = cfg.DATA.IMG_SIZE 
        self.num_slots = cfg.MODEL.SLOT.NUM
        self.image_dir = os.path.join(data_dir, 'images', self.phase)
        self.scene_dir = os.path.join(data_dir, 'scenes')
        
        self.files = sorted(os.listdir(self.image_dir))
        self.len_files = len(self.files)

        self.crop_size = cfg.DATA.CROP_SIZE 
        if self.crop_size > 0:
            self.img_train_transform = transforms.Compose([
                transforms.CenterCrop(self.crop_size),
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor()])
            self.img_val_transform = transforms.Compose([
                transforms.CenterCrop(self.crop_size),
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor()])
        else:
            self.img_train_transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor()])
            self.img_val_transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor()])
    
        self.center_crop = transforms.CenterCrop(self.crop_size)
        self.mask_resize = transforms.Resize((self.img_size, self.img_size))

        self.shapes = ['cube', 'sphere', 'cylinder', 'monkey']
        self.shape_to_idx = dict()
        for i, shape in enumerate(self.shapes): 
            self.shape_to_idx[shape] = i

    def __getitem__(self, index):
        index = index % self.len_files

        image_name = self.files[index]
        image = Image.open(os.path.join(self.image_dir, image_name)).convert("RGB") 
        w, h = image.size

        scene_name = image_name[:-3] + 'json'
        metadata = json.load(open(os.path.join(self.scene_dir, self.phase, scene_name)))

        image = self.img_train_transform(image)
        sample = {'image': image}

        '''
        coords = (x, y)
        shape = # ['cube', 'sphere', 'cylinder', 'monkey']
        real_obj = 0/1
        '''
        pixel_coords = torch.full((self.num_slots, 2), fill_value=-1, dtype=torch.float32)
        shape = torch.zeros((self.num_slots, len(self.shapes)), dtype=torch.long)
        real_obj = torch.zeros((self.num_slots, 1), dtype=torch.long)

        for obj_i in range(len(metadata['objects'])):
            pixel_coords[obj_i] = torch.tensor([ metadata['objects'][obj_i]['pixel_coords'][0] / w, metadata['objects'][obj_i]['pixel_coords'][1] / h ])
            shape[obj_i][self.shape_to_idx[metadata['objects'][obj_i]['shape']]] = 1
            real_obj[obj_i][0] = 1

        sample['pixel_coords'] = pixel_coords - 0.5
        sample['shape'] = shape
        sample['real_obj'] = real_obj

        return sample
        
    def __len__(self):
        return self.len_files
