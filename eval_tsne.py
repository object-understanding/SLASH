from datasets_sp import *
from model import *
import torch
import numpy as np
import argparse
import datetime
from tqdm import tqdm 
import pandas as pd 
import time

import matplotlib.pyplot as plt
from PIL import Image as Image, ImageEnhance
import seaborn as sns

from sklearn.manifold import TSNE


import torch.nn.functional as F

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

def process_predictions(predictions):
    ''' 
    Parameter:
        `predictions`: (N_slots, N_cls)
    Return: 
        `label`: (N_slots, 1)
    '''

    """Unpacks the target into the CLEVR properties."""
    coords = predictions[:, :2]
    object_size = torch.argmax(predictions[:, 2:4], dim=1)
    material = torch.argmax(predictions[:, 4:6], dim=1)
    shape = torch.argmax(predictions[:, 6:9], dim=1) 
    color = torch.argmax(predictions[:, 9:17], dim=1)
    real_obj = predictions[:, 17]
    real_obj = torch.where(real_obj > 0.5,
                           torch.ones_like(real_obj, dtype=torch.long, device=real_obj.device),
                           torch.zeros_like(real_obj, dtype=torch.long, device=real_obj.device))
    
    # TODO: handle coords 

    # labels = object_size * 10000 + material * 1000 + shape * 100 + color * 10 + real_obj
    # labels = object_size * 10000 + material * 1000 + shape * 100 + color * 10
    # labels = real_obj
    labels = object_size
    # labels = material
    # labels = shape
    # labels = color
    return labels[:, None]

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config_file', default='configs/config.yaml', type=str)
    parser.add_argument('--data_dir', default='/workspace/dataset/clevr_with_masks/CLEVR6', type=str)
    parser.add_argument('--batch_size', default=0, type=int)
    parser.add_argument('--num_workers', default=-1, type=int)
    parser.add_argument('--checkpoint', default=None)
    return parser

def main(args): 
    assert args.checkpoint is not None, "Wrong checkpoint!"
    
    checkpoint = args.checkpoint
    assert os.path.exists(checkpoint), "Wrong checkpoint!"

    cfg = set_config(args.config_file)
    if args.batch_size > 0:
        cfg.TRAIN.BATCH_SIZE = args.batch_size
    if args.num_workers > -1:
        cfg.DATA.NUM_WORKERS = args.num_workers
    cfg_str = '__'.join( ['{}={}'.format(k, v) for k, v in vars(cfg).items()] )

    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")

    model = SlotAttentionClassifier(cfg, device=device).to(device)
    # model = SlotAttentionAutoEncoder(cfg, device=device).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device)['model_state_dict'])
    params = [{'params': model.parameters()}]
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Device = %s" % device)
    print("Model = %s" % str(model))
    print('number of params (M) = %.2f' % (n_parameters / 1.e6))

    if cfg.DATA.TYPE in ['ptr', 'PTR']:
        dataset_val = PTR(data_dir=args.data_dir, phase='val', cfg=cfg)
    elif cfg.DATA.TYPE in ['clevr', 'CLEVR']:
        dataset_val = CLEVR(data_dir=args.data_dir, phase='val', cfg=cfg)
    elif cfg.DATA.TYPE.lower() == 'movi': 
        dataset_val = MOVi(data_dir=args.data_dir, phase='val', cfg=cfg)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, 
        pin_memory=True, 
        batch_size=cfg.TRAIN.BATCH_SIZE, 
        shuffle=False, 
        num_workers=cfg.DATA.NUM_WORKERS
    )

    B = cfg.TRAIN.BATCH_SIZE
    K = cfg.MODEL.SLOT.NUM
    D_slot = cfg.MODEL.SLOT.DIM
    D_mlp = cfg.MODEL.SET_PREDICTION.MLP_HID_DIM
    N_cls = cfg.MODEL.SET_PREDICTION.NUM_CLASSES

    with torch.no_grad(): 
        model.eval()
        for sample_i, sample in enumerate(tqdm(data_loader_val)):
            image = sample['image'].to(device)
            outputs = model(**dict(image=image, pos=None, train=False))
            if sample_i == 0:
                slots = outputs['slots'].reshape(B*K, D_slot)
                pred_slots = outputs['pred_slots'].reshape(B*K, D_mlp)
                predictions = outputs['predictions'].reshape(-1, N_cls)
            else: 
                slots = torch.cat([slots, outputs['slots'].reshape(-1, D_slot)], dim=0)
                pred_slots = torch.cat([pred_slots, outputs['pred_slots'].reshape(-1, D_mlp)], dim=0)
                predictions = torch.cat([predictions, outputs['predictions'].reshape(-1, N_cls)], dim=0)

        # `slots`: (N_slots, D_slots)
        # `predicions`: (N_slots, N_cls)

    labels = process_predictions(predictions).detach().cpu().numpy()
    n_labels = len(set(labels[:, 0]))
    print(f'Total {n_labels} labels')
    # `labels`: (N_slots, 1)

    start_tsne = time.time()
    # tsne_slots = TSNE(n_components=2, perplexity=50, n_iter=1000).fit_transform(slots.detach().cpu().numpy())
    tsne_slots = TSNE(n_components=2, perplexity=50, n_iter=1000).fit_transform(pred_slots.detach().cpu().numpy())
    end_tsne = time.time()
    elapsed_time = datetime.timedelta(seconds=int(end_tsne - start_tsne))
    print(f"T-SNE took {elapsed_time}")

    data = np.concatenate((tsne_slots, labels), axis=1)
    df_tsne_slots = pd.DataFrame(data, columns=["x", "y", "label"])

    start_plot = time.time()
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="x", 
        y="y", 
        hue="label",
        palette=sns.color_palette("hls", n_labels),
        # palette=sns.color_palette("husl", n_labels),
        data=df_tsne_slots, 
        legend=False,
        alpha=0.3
    )
    end_plot = time.time()
    elapsed_time = datetime.timedelta(seconds=int(end_plot - start_plot))
    print(f"Plotting took {elapsed_time}")

    filename = ".".join(checkpoint.split('.')[:-1]) + '_tsne_' + datetime.datetime.now().strftime("%y%m%d_%H%M%S") + '.png'
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    plt.savefig(filename)
    print('Figure is saved in', filename)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser('Slot Attention visualization script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)