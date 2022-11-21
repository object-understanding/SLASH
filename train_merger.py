import argparse
import datetime
import os
import time
import yaml
from pathlib import Path

import torch
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from datasets import *
from model import *
from utils.vutil import visualize
from utils.evaluator import ARIEvaluator
from utils.config import *

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config_file', default='configs/config.yaml', type=str)
    parser.add_argument('--data_dir', default='/workspace/dataset/clevr_with_masks/CLEVR6', type=str)
    parser.add_argument('--batch_size', default=0, type=int, help='Desired batch_size: 64 x num_gpus')
    parser.add_argument('--num_workers', default=-1, type=int)
    parser.add_argument('--output_dir_suffix', default='', type=str)
    parser.add_argument('--resume_ckpt', default='', type=str)
    return parser

def main(args):
    cfg = set_config(args.config_file)

    if args.batch_size > 0:
        cfg.TRAIN.BATCH_SIZE = args.batch_size
    if args.num_workers > -1:
        cfg.DATA.NUM_WORKERS = args.num_workers
    if cfg.OUTPUT.DIR[-1] == '/':
        cfg.OUTPUT.DIR = f"{cfg.OUTPUT.DIR[:-1]}_{args.output_dir_suffix}"
    else:
        cfg.OUTPUT.DIR = f"{cfg.OUTPUT.DIR}_{args.output_dir_suffix}"
    cfg_str = '__'.join( ['{}={}'.format(k, v) for k, v in vars(cfg).items()] )

    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    print(device, torch.cuda.device_count())

    if cfg.OUTPUT.DIR is not None:
        Path(cfg.OUTPUT.DIR).mkdir(parents=True, exist_ok=True)
        log_writer = SummaryWriter(log_dir=cfg.OUTPUT.DIR)
        log_writer.add_text('hparams', cfg_str)

    if cfg.DATA.TYPE in ['ptr', 'PTR']:
        dataset_train = PTR(data_dir=args.data_dir, phase='train', cfg=cfg)
        dataset_val = PTR(data_dir=args.data_dir, phase='val', cfg=cfg)
    elif cfg.DATA.TYPE in ['clevr', 'CLEVR']:
        dataset_train = CLEVR(data_dir=args.data_dir, phase='train', cfg=cfg)
        dataset_val = CLEVR(data_dir=args.data_dir, phase='val', cfg=cfg)
    elif cfg.DATA.TYPE in ['msn', 'MSN']: 
        dataset_train = MSN(data_dir=args.data_dir, phase='train', cfg=cfg)
        dataset_val = MSN(data_dir=args.data_dir, phase='val', cfg=cfg)

    # dataset_train = torch.utils.data.Subset(dataset_train, list(range(0, 100))) # for short training epoch
    # dataset_val = torch.utils.data.Subset(dataset_val, list(range(0, 100))) # for short val epoch

    loader_kwargs = {
        'batch_size': cfg.TRAIN.BATCH_SIZE,
        'shuffle': True,
        'num_workers': cfg.DATA.NUM_WORKERS,
        'pin_memory': True,
        'drop_last': True,
    }

    # data_loader_train = DataLoader(dataset_train, **loader_kwargs)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.DATA.NUM_WORKERS
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=cfg.DATA.NUM_WORKERS
    )

    model = SlotMerger(cfg)
    model = nn.DataParallel(model).to(device)
    print(model)

    criterion = nn.MSELoss()
    params = [{'params': model.parameters()}]
    optimizer = optim.Adam(params, lr=cfg.TRAIN.BASE_LR)

    if args.resume_ckpt != '': 
        assert os.path.exists(args.resume_ckpt), "Wrong checkpoint!"

        checkpoint = torch.load(args.resume_ckpt)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint['epoch']
        cfg.TRAIN.EPOCHS = checkpoint['epochs'] # optional?
        total_step = checkpoint['total_step']
        total_steps = checkpoint['total_steps'] # optional?
        cfg.TRAIN.WARMUP_STEP_RATIO = checkpoint['warmup_step_ratio']
        cfg.TRAIN.DECAY_STEP_RATIO = checkpoint['decay_step_ratio']
        cfg.TRAIN.DECAY_RATE = checkpoint['decay_rate']

        # elapsed time to train the checkpoint
        elapsed_time = checkpoint['elapsed_time']

    else:
        epoch = 0
        total_step = 0    
        total_steps = cfg.TRAIN.EPOCHS * len(data_loader_train)

        elapsed_time = 0

    warmup_steps = total_steps * cfg.TRAIN.WARMUP_STEP_RATIO
    decay_steps = total_steps * cfg.TRAIN.DECAY_STEP_RATIO

    for epoch in range(epoch, cfg.TRAIN.EPOCHS):
    
        start_epoch = time.time()
        model.train()
        total_loss = 0
        total_recon_loss = 0

        for sample in tqdm(data_loader_train, desc='Epoch {}/{}'.format(epoch+1, cfg.TRAIN.EPOCHS)):
            total_step += 1

            if total_step < warmup_steps:
                lr = cfg.TRAIN.BASE_LR * (total_step / warmup_steps)
            else:
                lr = cfg.TRAIN.BASE_LR

            lr = lr * (cfg.TRAIN.DECAY_RATE ** (total_step / decay_steps))

            optimizer.param_groups[0]['lr'] = lr

            images = sample['image'].to(device)
            
            outputs = model(images, init_slot_ratio=0.01,
                            init_patch_size=32, num_pairs_per_patch=2)

            recon_combined = outputs['recon_combined']
            loss = criterion(recon_combined, images)
            total_recon_loss += loss.item()
            total_loss += loss.item() 

            del outputs['recons'], outputs['masks'], outputs['slots'], outputs['attn']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = total_loss / len(data_loader_train)
        train_recon_loss = total_recon_loss / len(data_loader_train)

        end_epoch = time.time()
        elapsed_time += end_epoch - start_epoch

        print(
            "Epoch: {}, Total Loss: {:.3e}, Recon Loss: {:.3e}, Time: {}".format(
                epoch+1, 
                train_loss, 
                train_recon_loss,
                datetime.timedelta(seconds=int(elapsed_time))
            )
        )

        if log_writer is not None:
            print('log_dir: {}\n'.format(log_writer.log_dir))
            log_writer.add_scalar('train_loss', train_loss, epoch+1)
            log_writer.add_scalar('lr', lr, epoch+1)

        if not (epoch + 1) % 100 or (epoch + 1) == cfg.TRAIN.EPOCHS:

            checkpoint = {
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),

                'epoch': epoch + 1,
                'epochs': cfg.TRAIN.EPOCHS, 
                'total_step': total_step, 
                'total_steps': total_steps, 
                'warmup_step_ratio': cfg.TRAIN.WARMUP_STEP_RATIO,
                'decay_step_ratio': cfg.TRAIN.DECAY_STEP_RATIO,
                'decay_rate': cfg.TRAIN.DECAY_RATE,
                'elapsed_time': elapsed_time
                # 'args': args

                # 'best_val_loss': best_val_loss,
                # 'best_epoch': best_epoch,
            }

            torch.save(
                checkpoint,
                os.path.join(cfg.OUTPUT.DIR, f'checkpoint-{epoch+1}.pth'),
            )
            if cfg.OUTPUT.SAVE_SLOT_MEMORY and cfg.MODEL.SLOT_MEMORY.USE_MEMORY:
                torch.save(
                    model.module.slot_memory,
                    os.path.join(cfg.OUTPUT.DIR, f'slot_memory-{epoch+1}.pt'),
                )

        # evaluation
        if (epoch+1) % cfg.TRAIN.EVAL_INTERVAL == 0:
            with torch.no_grad(): 
                start_epoch = time.time()
                
                model.eval()
                total_val_loss = 0
                total_val_recon_loss = 0
                ari_evaluator = ARIEvaluator()
                for sample in tqdm(data_loader_val, desc='Val {}/{}'.format(epoch+1, cfg.TRAIN.EPOCHS)):
                    images = sample['image'].to(device)
                    outputs = model(images, init_slot_ratio=0.25,
                                    init_patch_size=32, num_pairs_per_patch=2)
                    
                    recon_combined = outputs['recon_combined']
                    loss = criterion(recon_combined, images)
                    total_val_recon_loss += loss.item()
                    total_val_loss += loss.item()

                    masks = outputs['masks']
                    ari_evaluator.evaluate(masks, sample['masks'], device)

                val_loss = total_val_loss / len(data_loader_val)
                val_recon_loss = total_val_recon_loss / len(data_loader_val)
                val_ari = ari_evaluator.get_results()

                end_epoch = time.time()
                print(
                    "Epoch: {}/{}, ARI: {:.4f}, Total Loss: {:.3e}, Recon Loss: {:.3e}, Time: {}".format(
                        epoch+1, cfg.TRAIN.EPOCHS, 
                        val_ari,
                        val_loss,
                        val_recon_loss, 
                        datetime.timedelta(seconds=end_epoch - start_epoch)
                    )
                )
                if log_writer is not None:
                    print('log_dir: {}\n'.format(log_writer.log_dir))
                    log_writer.add_scalar('val_loss', val_loss, epoch+1)
                    log_writer.add_scalar('val_ari', val_ari, epoch+1)

                    recons = outputs['recons']
                    log_img = visualize(images, recon_combined, recons, masks, N=8)
                    log_img = vutils.make_grid(log_img, nrow=1, pad_value=0)
                    log_writer.add_image('val_visualization/epoch={:04}'.format(epoch+1), log_img)

    log_writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Slot Attention training script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
