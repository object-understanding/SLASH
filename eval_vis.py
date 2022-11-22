import os 
from concurrent.futures import thread
from datasets import *
from model import *
import torch
import numpy as np
import argparse
from datetime import datetime

import matplotlib.pyplot as plt
from PIL import Image as Image, ImageEnhance

from sklearn.metrics.cluster import adjusted_rand_score

import torch.nn.functional as F
import torchvision.utils as vutils

from utils.config import *
from utils.vutil import *

class Config(object): 

    def __init__(self, config):
        self.config_file = config
        try:
            with open(self.config_file) as config_file:
                config = yaml.safe_load(config_file)
                for k in config.keys():
                    setattr(self, k, config[k])
        except Exception as e: 
            print(e)
            print('Wrong config file!')


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config_file', default='configs/config.yaml', type=str)
    parser.add_argument('--data_dir', default='/workspace/dataset/clevr_with_masks/CLEVR6', type=str)
    parser.add_argument('--num_workers', default=-1, type=int)
    parser.add_argument('--workspace', default='.')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--num_vis', default=1, type=int)
    parser.add_argument('--use_weak_sup', action='store_true')
    parser.add_argument('--tag', default='', type=str)
    parser.add_argument('--idx', default=-1, type=int)
    parser.add_argument('--idx_list', default='', type=str)
    parser.add_argument('--vis_type', default='normal', type=str)
    return parser

def main(args): 
    assert args.checkpoint is not None, "Wrong checkpoint!"
    assert os.path.exists(args.checkpoint), "Wrong checkpoint!"
    checkpoint = args.checkpoint
    workspace = args.workspace
    os.makedirs(workspace, exist_ok=True)

    cfg = set_config(args.config_file)
    if args.num_workers > -1:
        cfg.DATA.NUM_WORKERS = args.num_workers
    cfg_str = '__'.join( ['{}={}'.format(k, v) for k, v in vars(cfg).items()] )

    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")

    model = SlotAttentionAutoEncoder(cfg, device).to(device)
    # model = nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device)['model_state_dict'])

    # print(model.slot_attention.knconv.weight); exit()
    if cfg.DATA.TYPE.lower() == 'ptr':
        dataset = PTR(data_dir=args.data_dir, phase='val', cfg=cfg)
    elif cfg.DATA.TYPE.lower() == 'clevr':
        dataset = CLEVR(data_dir=args.data_dir, phase='val', cfg=cfg)
    elif cfg.DATA.TYPE.lower() == 'clevrtex':
        dataset = CLEVRTEX(data_dir=args.data_dir, phase='val', cfg=cfg)
    elif cfg.DATA.TYPE.lower() == 'movi':
        dataset = MOVi(data_dir=args.data_dir, phase='val', cfg=cfg)

    if args.idx_list != '':
        idx_list = np.array(list(map(int, args.idx_list.split(','))))
        idx_list = np.repeat(idx_list, args.num_vis)
    else:
        if args.idx > -1: 
            idx_list = np.array([args.idx] * args.num_vis)
        else:
            idx_list = np.random.randint(len(dataset.files), size=args.num_vis)
        
    for i, idx in enumerate(idx_list):
        img_list = [dataset[idx]['image']]
        mask_list = [dataset[idx]['masks']]
        if cfg.WEAK_SUP.TYPE != '':
            pos_list = [cfg.WEAK_SUP.TYPE]   

        # make it a batch-like
        image = torch.stack(img_list, dim=0).to(cfg.DEVICE)
        masks = torch.stack(mask_list, dim=0).to(cfg.DEVICE)
        if cfg.WEAK_SUP.TYPE != '' and args.use_weak_sup:
            pos = torch.stack(pos_list, dim=0).to(cfg.DEVICE)
        else:
            pos = None
        print("Use position gt:", cfg.WEAK_SUP.TYPE != '' and args.use_weak_sup)

        # run model
        with torch.no_grad():
            outputs = model(image, pos, train=False)
        attns = torch.stack(outputs['attns'], dim=1)
        attns_origin = torch.stack(outputs['attns_origin'], dim=1)
        if cfg.POS_PRED.USE_POS_PRED:
            pos_pred = torch.stack(outputs['pos_pred'], dim=1)
            # `pos_pred_origin`: (B, L, K, 2)

            # omit the position prediction for the 0-th iteration
            if 0 in cfg.POS_PRED.LOCATIONS: 
                pos_pred = pos_pred[:, 1:]
        else: 
            pos_pred = None 
        
        filename  = datetime.now().strftime("%y%m%d_%H%M%S")+ '_idx' + str(idx)
        if args.tag != '':  
            filename = filename + '_' + args.tag
        filename = filename + '_' + str(i)

        if args.vis_type == 'normal':
            log_img = visualize(image=image, 
                                recon_combined=outputs['recon_combined'],
                                recons=outputs['recons'], 
                                masks=outputs['masks'], 
                                attns=attns, 
                                pos_pred=pos_pred, # pos_pred=None, 
                                pos_pred_loc=cfg.POS_PRED.LOCATIONS)

        if args.vis_type == 'attns':
            log_img = visualize_attns(image=image, 
                                    recon_combined=outputs['recon_combined'],
                                    recons=outputs['recons'], 
                                    pred_masks=outputs['masks'], 
                                    gt_masks=masks.unsqueeze(-1),
                                    attns=attns, 
                                    # attns_origin=None, 
                                    attns_origin=attns_origin,
                                    pos_pred=None, 
                                    pos_pred_loc=None)
                                    
        log_img = vutils.make_grid(log_img, nrow=1, pad_value=0)
        log_img = log_img.permute(1, 2, 0)
        plt. figure(figsize = (20, 10))
        plt.tight_layout()
        plt.imshow(log_img.detach().cpu().numpy())
        plt.axis('off')
        plt.savefig(os.path.join(workspace, filename))
        plt.cla()

if __name__ == '__main__': 
    parser = argparse.ArgumentParser('Slot Attention visualization script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
