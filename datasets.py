import os
import math
import random
import json
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn.functional as F 
from pycocotools import mask
from collections import defaultdict


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
        self.mask_dir = os.path.join(data_dir, 'masks', self.phase)
        self.scene_dir = os.path.join(data_dir, 'scenes')
        self.metadata = json.load(open(os.path.join(self.scene_dir, f"CLEVR_{self.phase}_scenes.json")))
        
        self.ws_split_ratio = cfg.WEAK_SUP.SPLIT.RATIO
        if self.sub:
            split_data_dir = os.path.join(data_dir, 'supervision_splits.json')
            split_data = json.load(open(split_data_dir, 'r'))
            self.files = sorted(split_data[str(self.ws_split_ratio)]['known'])
        else:
            self.files = sorted(os.listdir(self.image_dir))
        self.len_files = len(self.files)

        self.weak_supervision = cfg.WEAK_SUP.TYPE
        self.ws_random_prop = cfg.WEAK_SUP.RAND_PROP

        self.use_batch_fusion = cfg.WEAK_SUP.SPLIT.TRAIN.BATCH_FUSION
        if self.use_batch_fusion:
            self.batch_size = cfg.TRAIN.BATCH_SIZE
            self.batch_fusion_ratio = cfg.WEAK_SUP.SPLIT.TRAIN.BATCH_FUSION_RATIO
                
        self.masks = defaultdict(list)
        masks = sorted(os.listdir(self.mask_dir))
        for mask in masks: 
            split =  mask.split('_')
            filename = '_'.join(split[:3]) + '.png'
            self.masks[filename].append(mask)

        del masks 

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


    def __getitem__(self, index):
        index = index % self.len_files
        
        filename = self.metadata['scenes'][index]['image_filename']
        while filename not in self.files:
            index = random.randint(0, len(self.files))
            filename = self.metadata['scenes'][index]['image_filename']

        image = Image.open(os.path.join(self.image_dir, filename)).convert("RGB")
        w, h = image.size

        if self.phase == 'train':
            image = self.img_train_transform(image)
            sample = {'image': image}
        elif self.phase == 'val': 
            image = self.img_val_transform(image)
            masks = [ (transforms.functional.pil_to_tensor(Image.open(os.path.join(self.mask_dir, mask_filename)).convert("L")) // 255).long()
                      for mask_filename in self.masks[filename] ]
            masks = torch.stack(masks, dim=0) # (N + 1, 1, H, W)
            if self.crop_size > 0:
                masks = self.center_crop(masks) 
            masks = self.mask_resize(masks).squeeze(1) # (N + 1, H, W)
            n_masks = masks.shape[0]
            if n_masks < self.num_slots:
                masks = torch.cat((masks, torch.zeros((self.num_slots - n_masks, self.img_size, self.img_size))), dim=0)
            sample = {'image': image, 'masks': masks.float()}
        
        if self.weak_supervision != "":
            bbox = torch.tensor([[-1., -1., -1., -1.] for _ in range(self.num_slots)], requires_grad=False)
            bbox_center = torch.tensor([[-1., -1.] for _ in range(self.num_slots)], requires_grad=False)
            bbox_gt = torch.tensor([[-1., -1., -1., -1.] for _ in range(self.num_slots)], requires_grad=False)
            bbox_center_gt = torch.tensor([[-1., -1.] for _ in range(self.num_slots)], requires_grad=False)
            for obj_i in range(1, len(self.metadata['scenes'][index]['objects'])):
                gt_bbox = self.metadata['scenes'][index]['objects'][obj_i]['bbox']
                gt_pixel_coords = self.metadata['scenes'][index]['objects'][obj_i]['pixel_coords']
                if np.random.random() < self.ws_random_prop:
                    bbox[obj_i] = torch.tensor([gt_bbox[0] / w, gt_bbox[1] / w, gt_bbox[2] / h, gt_bbox[3] / h], requires_grad=False)
                    bbox_center[obj_i] =  torch.tensor([gt_pixel_coords[0] / w, gt_pixel_coords[1] / h], requires_grad=False)
                bbox_gt[obj_i] = torch.tensor([gt_bbox[0] / w, gt_bbox[1] / w, gt_bbox[2] / h, gt_bbox[3] / h], requires_grad=False)
                bbox_center_gt[obj_i] =  torch.tensor([gt_pixel_coords[0] / w, gt_pixel_coords[1] / h], requires_grad=False)
            # pos range -0.5 ~ 0.5 (to match with gaussian rand_pos and etc)
            sample['bbox'] = bbox - 0.5
            sample['bbox_center'] = bbox_center - 0.5
            sample['bbox_gt'] = bbox_gt - 0.5
            sample['bbox_center_gt'] = bbox_center_gt - 0.5

        return sample
        
    def __len__(self):
        # make sub dataset longer
        # to prevent early stop in dataloader when zip original and sub loader
        if self.sub and self.use_batch_fusion:
            repeat_num = math.ceil(1 / self.ws_split_ratio)
            repeat_num = math.ceil(repeat_num / ((1 - self.batch_fusion_ratio) / self.batch_fusion_ratio))
            return self.len_files * repeat_num
        else:
            return self.len_files


class CLEVRTEX(Dataset):
    def __init__(self, data_dir, phase='train', sub=False, cfg=None):
        super(CLEVRTEX, self).__init__()
        assert phase in ['train', 'val']
        assert cfg.WEAK_SUP.TYPE in ['', 'bbox_center']
        assert not sub or (sub and cfg.WEAK_SUP.SPLIT.RATIO < 1)

        self.sub = sub
        self.phase = phase
        self.img_size = cfg.DATA.IMG_SIZE
        self.num_slots = cfg.MODEL.SLOT.NUM
        self.image_dir = os.path.join(data_dir, 'images', self.phase)
        self.mask_dir = os.path.join(data_dir, 'masks', self.phase)
        self.scene_dir = os.path.join(data_dir, 'scenes')
        
        self.ws_split_ratio = cfg.WEAK_SUP.SPLIT.RATIO
        if self.sub:
            split_data_dir = os.path.join(data_dir, 'supervision_splits.json')
            split_data = json.load(open(split_data_dir, 'r'))
            self.files = sorted(split_data[str(self.ws_split_ratio)]['known'])
        else:
            self.files = sorted(os.listdir(self.image_dir))
        self.len_files = len(self.files)

        self.weak_supervision = cfg.WEAK_SUP.TYPE
        self.ws_random_prop = cfg.WEAK_SUP.RAND_PROP

        self.use_batch_fusion = cfg.WEAK_SUP.SPLIT.TRAIN.BATCH_FUSION
        if self.use_batch_fusion:
            self.batch_size = cfg.TRAIN.BATCH_SIZE
            self.batch_fusion_ratio = cfg.WEAK_SUP.SPLIT.TRAIN.BATCH_FUSION_RATIO

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


    def __getitem__(self, index):
        index = index % self.len_files

        image_name = self.files[index]
        image = Image.open(os.path.join(self.image_dir, image_name)).convert("RGB") 
        w, h = image.size

        scene_name = image_name[:-3] + 'json'
        metadata = json.load(open(os.path.join(self.scene_dir, self.phase, scene_name)))

        if self.phase == 'train':
            image = self.img_train_transform(image)
            sample = {'image': image}
        elif self.phase == 'val': 
            image = self.img_val_transform(image)
            mask_name = image_name[:-4] + "_flat.png"
            gt_masks = transforms.functional.pil_to_tensor(Image.open(os.path.join(self.mask_dir, mask_name))).squeeze(0).long() 
            gt_masks = F.one_hot(gt_masks, len(metadata['objects']) + 1).permute(2, 0, 1)
            if self.crop_size > 0:
                gt_masks = self.center_crop(gt_masks) 
            gt_masks = self.mask_resize(gt_masks)
            # `gt_masks`: (N+1, H, W)

            n_masks = gt_masks.shape[0]
            if n_masks < self.num_slots:
                gt_masks = torch.cat((gt_masks, torch.zeros((self.num_slots - n_masks, self.img_size, self.img_size))), dim=0)
        
            sample = {'image': image, 'masks': gt_masks.float()}

        if self.weak_supervision != "":

            # the label having only -1 means that no informantion is given
            bbox_center = torch.full((self.num_slots, 2), -1., requires_grad=False)
            bbox_center_gt = torch.full((self.num_slots, 2), -1., requires_grad=False)

            # for objects (due to the background, index begins with 1)
            for obj_i in range(len(metadata['objects'])):
                gt_bbox_center = metadata['objects'][obj_i]['pixel_coords'][:2] # except for the last element which is depth
                if np.random.random() < self.ws_random_prop:
                    bbox_center[obj_i] =  torch.tensor([ gt_bbox_center[0] / w, gt_bbox_center[1] / h ])
                bbox_center_gt[obj_i] =  torch.tensor([ gt_bbox_center[0] / w, gt_bbox_center[1] / h ])
            # pos range -0.5 ~ 0.5 (to match with gaussian rand_pos and etc)
            sample['bbox_center'] = bbox_center - 0.5
            sample['bbox_center_gt'] = bbox_center_gt - 0.5

        return sample
        
    def __len__(self):
        # make sub dataset longer
        # to prevent early stop in dataloader when zip original and sub loader

        if self.sub and self.use_batch_fusion:
            repeat_num = math.ceil(1 / self.ws_split_ratio)
            repeat_num = math.ceil(repeat_num / ((1 - self.batch_fusion_ratio) / self.batch_fusion_ratio))
            return self.len_files * repeat_num
        else:
            return self.len_files


class PTR(Dataset):
    def __init__(self, data_dir, phase='train', sub=False, cfg=None):
        super(PTR, self).__init__()
        
        assert phase in ['train', 'val', 'test']
        assert cfg.WEAK_SUP.TYPE in ['', 'bbox', 'bbox_center']
        assert not sub or (sub and cfg.WEAK_SUP.SPLIT.RATIO < 1)

        self.sub = sub
        self.phase = phase 
        self.img_size = cfg.DATA.IMG_SIZE
        self.num_slots = cfg.MODEL.SLOT.NUM
        self.image_dir = os.path.join(data_dir, 'images', self.phase)
        self.scene_dir = os.path.join(data_dir, 'scenes', self.phase)

        self.ws_split_ratio = cfg.WEAK_SUP.SPLIT.RATIO
        if self.sub:
            split_data_dir = os.path.join(data_dir, 'supervision_splits.json')
            split_data = json.load(open(split_data_dir, 'r'))
            self.files = sorted(split_data[str(self.ws_split_ratio)]['known'])
        else:
            self.files = sorted(os.listdir(self.image_dir))
        self.len_files = len(self.files)

        self.weak_supervision = cfg.WEAK_SUP.TYPE
        self.ws_random_prop = cfg.WEAK_SUP.RAND_PROP


        self.use_batch_fusion = cfg.WEAK_SUP.SPLIT.TRAIN.BATCH_FUSION
        if self.use_batch_fusion:
            self.batch_size = cfg.TRAIN.BATCH_SIZE
            self.batch_fusion_ratio = cfg.WEAK_SUP.SPLIT.TRAIN.BATCH_FUSION_RATIO
        
        self.img_train_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()])
        
        self.img_val_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()])
            
        self.mask_resize = transforms.Resize((self.img_size, self.img_size))

    def __getitem__(self, index):
        index = index % self.len_files

        image_name = self.files[index]
        image = Image.open(os.path.join(self.image_dir, image_name)).convert("RGB")    
        w, h = image.size

        scene_name = image_name[:-3] + 'json'
        metadata = json.load(open(os.path.join(self.scene_dir, scene_name)))

        if self.phase == 'train':
            image = self.img_train_transform(image)
            sample = {'image': image}

        elif self.phase == 'val': 
            try:
                image = self.img_val_transform(image)
                gt_masks = [] 
                for obj in metadata['objects']:
                    gt_masks.append(obj['obj_mask']) 
                gt_masks = torch.tensor(mask.decode(gt_masks), dtype=torch.long)
                gt_masks = torch.einsum('hwn -> nhw', gt_masks)
                gt_masks = self.mask_resize(gt_masks.unsqueeze(1)).squeeze(1)
                gt_masks = torch.cat([(torch.sum(gt_masks, dim=0, keepdim=True) == 0).long(), gt_masks], dim=0)
                # gt_masks: [n_objs + 1, H, W]

                # for multi-batch processing, pad dummy masks.
                # this may affect the performance measuremnets.
                n_masks = gt_masks.shape[0] 
                if n_masks < self.num_slots:
                    gt_masks = torch.cat((gt_masks, torch.zeros((self.num_slots - n_masks, self.img_size, self.img_size))), dim=0)

                sample = {'image': image, 'masks': gt_masks.float()}    

            except Exception as e:
                print(e)

        if self.weak_supervision:
            # the label having only -1 means that no informantion is given
            bbox = torch.full((self.num_slots, 4), -1.)
            bbox_center = torch.full((self.num_slots, 2), -1.)
            bbox_gt = torch.full((self.num_slots, 4), -1.)
            bbox_center_gt = torch.full((self.num_slots, 2), -1.)

            # for background 
            bbox[0] = torch.tensor([0, 0, 0, 0])
            bbox_center[0] =  torch.tensor([0, 0])
            bbox_gt[0] = torch.tensor([0, 0, 0, 0])
            bbox_center_gt[0] =  torch.tensor([0, 0])

            for obj_i in range(len(metadata['objects'])):
                gt_bbox = metadata['objects'][obj_i]['bbox']
                if np.random.random() < self.ws_random_prop:
                    bbox[obj_i] = torch.tensor([gt_bbox[0] / w, gt_bbox[1] / w, gt_bbox[2] / h, gt_bbox[3] / h], requires_grad=False)
                    bbox_center[obj_i] =  torch.tensor([ (gt_bbox[0] + gt_bbox[1]) / 2 / w, (gt_bbox[2] + gt_bbox[3]) / 2/ h ], requires_grad=False)
                bbox_gt[obj_i] = torch.tensor([gt_bbox[0] / w, gt_bbox[1] / w, gt_bbox[2] / h, gt_bbox[3] / h], requires_grad=False)
                bbox_center_gt[obj_i] =  torch.tensor([ (gt_bbox[0] + gt_bbox[1]) / 2 / w, (gt_bbox[2] + gt_bbox[3]) / 2/ h ], requires_grad=False)
            # pos range -0.5 ~ 0.5 (to match with gaussian rand_pos and etc)
            sample['bbox'] = bbox - 0.5
            sample['bbox_center'] = bbox_center - 0.5
            sample['bbox_gt'] = bbox_gt - 0.5
            sample['bbox_center_gt'] = bbox_center_gt - 0.5
        
        return sample
        
    def __len__(self):
        # make sub dataset longer
        # to prevent early stop in dataloader when zip original and sub loader
        if self.sub and self.use_batch_fusion:
            repeat_num = math.ceil(1 / self.ws_split_ratio)
            repeat_num = math.ceil(repeat_num / ((1 - self.batch_fusion_ratio) / self.batch_fusion_ratio))
            return self.len_files * repeat_num
        else:
            return self.len_files


class MOVi(Dataset):
    def __init__(self, data_dir, phase='train', sub=False, cfg=None):
        super(MOVi, self).__init__()
        assert phase in ['train', 'val', 'val_from_train']
        assert cfg.WEAK_SUP.TYPE in ['', 'bbox', 'bbox_center']
        assert not sub or (sub and cfg.WEAK_SUP.SPLIT.RATIO < 1)

        self.sub = sub
        self.phase = phase
        self.img_size = cfg.DATA.IMG_SIZE
        self.num_slots = cfg.MODEL.SLOT.NUM
        self.image_dir = os.path.join(data_dir, 'images', self.phase)
        self.mask_dir = os.path.join(data_dir, 'masks', self.phase)
        self.scene_dir = os.path.join(data_dir, 'scenes')
        
        self.ws_split_ratio = cfg.WEAK_SUP.SPLIT.RATIO
        if self.sub:
            split_data_dir = os.path.join(data_dir, 'supervision_splits.json')
            split_data = json.load(open(split_data_dir, 'r'))
            self.files = sorted(split_data[str(self.ws_split_ratio)]['known'])
        else:
            self.files = sorted(os.listdir(self.image_dir))
        self.len_files = len(self.files)

        self.weak_supervision = cfg.WEAK_SUP.TYPE
        self.ws_random_prop = cfg.WEAK_SUP.RAND_PROP

        self.use_batch_fusion = cfg.WEAK_SUP.SPLIT.TRAIN.BATCH_FUSION
        if self.use_batch_fusion:
            self.batch_size = cfg.TRAIN.BATCH_SIZE
            self.batch_fusion_ratio = cfg.WEAK_SUP.SPLIT.TRAIN.BATCH_FUSION_RATIO

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


    def __getitem__(self, index):
        index = index % self.len_files

        image_name = self.files[index]
        image = Image.open(os.path.join(self.image_dir, image_name)).convert("RGB") 
        w, h = image.size

        scene_name = image_name[:-3] + 'json'
        metadata = json.load(open(os.path.join(self.scene_dir, self.phase, scene_name)))

        if self.phase == 'train':
            image = self.img_train_transform(image)
            sample = {'image': image}
        elif self.phase == 'val' or self.phase == 'val_from_train': 
            image = self.img_val_transform(image)
            gt_masks = transforms.functional.pil_to_tensor(Image.open(os.path.join(self.mask_dir, image_name))).squeeze(0).long() 
            gt_masks = F.one_hot(gt_masks, len(metadata['instances']) + 1).permute(2, 0, 1)
            if self.crop_size > 0:
                gt_masks = self.center_crop(gt_masks) 
            gt_masks = self.mask_resize(gt_masks)
            # `gt_masks`: (N+1, H, W)

            n_masks = gt_masks.shape[0]
            if n_masks < self.num_slots:
                gt_masks = torch.cat((gt_masks, torch.zeros((self.num_slots - n_masks, self.img_size, self.img_size))), dim=0)
        
            sample = {'image': image, 'masks': gt_masks.float()}

        if self.weak_supervision != "":

            # the label having only -1 means that no informantion is given
            bbox = torch.full((self.num_slots, 4), -1., requires_grad=False)
            bbox_center = torch.full((self.num_slots, 2), -1., requires_grad=False)
            bbox_gt = torch.full((self.num_slots, 4), -1., requires_grad=False)
            bbox_center_gt = torch.full((self.num_slots, 2), -1., requires_grad=False)

            # for objects (due to the background, index begins with 1)
            for obj_i in range(len(metadata['instances'])):
                '''
                bboxes: (None, 4) [float32]
                    The normalized image-space (2D) coordinates of the bounding box
                    [ymin, xmin, ymax, xmax] for all the frames in which the object is visible
                    (as specified in bbox_frames).
                ''' 
                gt_bbox = metadata['instances'][obj_i]['bboxes'][0]
                if np.random.random() < self.ws_random_prop:
                    bbox[obj_i] = torch.tensor([gt_bbox[1], gt_bbox[3], gt_bbox[0], gt_bbox[2]])
                    bbox_center[obj_i] =  torch.tensor([ (gt_bbox[1] + gt_bbox[3]) / 2, (gt_bbox[0] + gt_bbox[2]) / 2 ])
                bbox_gt[obj_i] = torch.tensor([gt_bbox[1], gt_bbox[3], gt_bbox[0], gt_bbox[2]])
                bbox_center_gt[obj_i] =  torch.tensor([ (gt_bbox[1] + gt_bbox[3]) / 2, (gt_bbox[0] + gt_bbox[2]) / 2 ])
            # pos range -0.5 ~ 0.5 (to match with gaussian rand_pos and etc)
            sample['bbox'] = bbox - 0.5
            sample['bbox_center'] = bbox_center - 0.5
            sample['bbox_gt'] = bbox_gt - 0.5
            sample['bbox_center_gt'] = bbox_center_gt - 0.5

        return sample
        
    def __len__(self):
        # make sub dataset longer
        # to prevent early stop in dataloader when zip original and sub loader

        if self.sub and self.use_batch_fusion:
            repeat_num = math.ceil(1 / self.ws_split_ratio)
            repeat_num = math.ceil(repeat_num / ((1 - self.batch_fusion_ratio) / self.batch_fusion_ratio))
            return self.len_files * repeat_num
        else:
            return self.len_files
