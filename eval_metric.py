import os 
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from datasets import *
from model import *
from utils.evaluator import ARIEvaluator, mIoUEvaluator
from utils.config import *

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
    parser.add_argument('--data_dir', default='data/clevr_with_masks/CLEVR6', type=str)
    parser.add_argument('--batch_size', default=0, type=int)
    parser.add_argument('--num_workers', default=-1, type=int)
    parser.add_argument('--output_dir', default='.')
    parser.add_argument('--checkpoint', default=None)

    return parser

def main(args): 
    assert args.checkpoint is not None, "Wrong checkpoint!"
    assert os.path.exists(args.checkpoint), "Wrong checkpoint!"
    checkpoint = args.checkpoint
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    cfg = set_config(args.config_file)
    if args.batch_size > 0:
        cfg.TRAIN.BATCH_SIZE = args.batch_size
    if args.num_workers > -1:
        cfg.DATA.NUM_WORKERS = args.num_workers
    cfg_str = '__'.join( ['{}={}'.format(k, v) for k, v in vars(cfg).items()] )

    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    print(device, torch.cuda.device_count())

    if cfg.DATA.TYPE.lower() == 'ptr':
        dataset_val = PTR(data_dir=args.data_dir, phase='val', cfg=cfg)
    elif cfg.DATA.TYPE.lower() == 'clevr':
        dataset_val = CLEVR(data_dir=args.data_dir, phase='val', cfg=cfg)
    elif cfg.DATA.TYPE.lower() == 'clevrtex':
        dataset_val = CLEVRTEX(data_dir=args.data_dir, phase='val', cfg=cfg)
    elif cfg.DATA.TYPE.lower() == 'movi':
        dataset_val = MOVi(data_dir=args.data_dir, phase='val', cfg=cfg)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=cfg.DATA.NUM_WORKERS
    )
                            
    model = SlotAttentionAutoEncoder(cfg, device).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device)['model_state_dict'])

    f_ari_evaluator = ARIEvaluator()
    ari_evaluator = ARIEvaluator()
    f_miou_evaluator = mIoUEvaluator()
    miou_evaluator = mIoUEvaluator()

    criterion = nn.MSELoss()
    total_mse = 0 

    model.eval()
    with torch.no_grad(): 
        for sample in tqdm(data_loader_val): 
            image = sample['image'].to(device)
            outputs = model(**dict(image=image, pos=None, train=False))

            recon_combined = outputs['recon_combined']
            masks = outputs['masks']

            f_ari_evaluator.evaluate(masks, sample['masks'][:, 1:], device)
            ari_evaluator.evaluate(masks, sample['masks'], device)
            f_miou_evaluator.evaluate(masks, sample['masks'][:, 1:], device)
            miou_evaluator.evaluate(masks, sample['masks'], device)
                
            total_mse += criterion(recon_combined, image).item()

        total_mse /= len(data_loader_val)
    
    f_ari_result = f_ari_evaluator.get_results()
    ari_result = ari_evaluator.get_results()
    f_miou_result = f_miou_evaluator.get_results()
    miou_result = miou_evaluator.get_results()

    print(args.checkpoint)
    print("MSE:", total_mse)
    print("FG-ARI:", f_ari_result)
    print("ARI:", ari_result)
    print("FG-mIoU:", f_miou_result)
    print("mIoU:", miou_result)
    

    with open(os.path.join(args.output_dir, 'eval_results.txt'), 'a') as f:
        f.write('Model: ' + args.checkpoint + '\n')
        f.write('MSE: {:.4f}\n'.format(total_mse))
        f.write('FG-ARI: {:.4f}\n'.format(f_ari_result))
        f.write('ARI: {:.4f}\n'.format(ari_result))
        f.write('FG-mIoU: {:.4f}\n'.format(f_miou_result))
        f.write('mIoU: {:.4f}\n'.format(miou_result))
        f.write('\n')

if __name__=="__main__": 
    parser = argparse.ArgumentParser('Slot Attention ARI calculation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
