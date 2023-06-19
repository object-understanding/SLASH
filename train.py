import copy
import argparse
import datetime
import os
import time
from pathlib import Path

import torch
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from datasets import *
from model import *
from utils.vutil import visualize
from utils.evaluator import ARIEvaluator, mIoUEvaluator
from utils.config import *

print("torch ver:", torch.__version__)

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config_file', default='configs/config.yaml', type=str)
    parser.add_argument('--data_dir', default='data/clevr_with_masks/CLEVR6', type=str)
    parser.add_argument('--batch_size', default=0, type=int, help='Desired batch_size: 64 x num_gpus')
    parser.add_argument('--lr', default=0, type=float)
    parser.add_argument('--eval_interval', default=0, type=int)
    parser.add_argument('--num_workers', default=-1, type=int)
    parser.add_argument('--output_dir', default='', type=str)
    parser.add_argument('--output_dir_suffix', default='', type=str)
    parser.add_argument('--resume_ckpt', default='', type=str)
    parser.add_argument('--epochs', default=0, type=int)
    parser.add_argument('--num_vis', default=4, type=int)

    return parser


def main(args):
    cfg = set_config(args.config_file)

    if args.batch_size > 0:
        cfg.TRAIN.BATCH_SIZE = args.batch_size
    if args.lr > 0:
        cfg.TRAIN.BASE_LR = args.lr
    if args.eval_interval > 0:
        cfg.TRAIN.EVAL_INTERVAL = args.eval_interval
    if args.num_workers > -1:
        cfg.DATA.NUM_WORKERS = args.num_workers
    if args.epochs > 0:
        cfg.TRAIN.EPOCHS = args.epochs
    if args.output_dir != '': 
        cfg.OUTPUT.DIR = args.output_dir
    if args.output_dir_suffix != '':
        if cfg.OUTPUT.DIR[-1] == '/':
            cfg.OUTPUT.DIR = f"{cfg.OUTPUT.DIR[:-1]}_{args.output_dir_suffix}"
        else:
            cfg.OUTPUT.DIR = f"{cfg.OUTPUT.DIR}_{args.output_dir_suffix}"
    cfg_str = '__'.join( ['{}={}'.format(k, v) for k, v in vars(cfg).items()] )

    use_amp = True

    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"{device}, {torch.cuda.device_count()}")

    if cfg.WEAK_SUP.SPLIT.TRAIN.BATCH_FUSION:
        use_batch_fusion = True
        batch_sufion_ratio = cfg.WEAK_SUP.SPLIT.TRAIN.BATCH_FUSION_RATIO
        batch_fusion_ws_num_samples = int(cfg.TRAIN.BATCH_SIZE * batch_sufion_ratio)
    else:
        use_batch_fusion = False

    if cfg.OUTPUT.DIR is not None: 
        Path(cfg.OUTPUT.DIR).mkdir(parents=True, exist_ok=True)
        log_writer = SummaryWriter(log_dir=cfg.OUTPUT.DIR)
        log_writer.add_text('hparams', cfg_str)
    else: 
        log_writer = None

    dataset_train_sub = []

    if cfg.DATA.TYPE.lower() == 'clevr':
        dataset_train = CLEVR(data_dir=args.data_dir, phase='train', cfg=cfg)
        dataset_val = CLEVR(data_dir=args.data_dir, phase='val', cfg=cfg)
        if cfg.WEAK_SUP.SPLIT.RATIO < 1 or use_batch_fusion:
            dataset_train_sub = CLEVR(data_dir=args.data_dir, phase='train', sub=True, cfg=cfg)

    elif cfg.DATA.TYPE.lower() == 'clevrtex':
        dataset_train = CLEVRTEX(data_dir=args.data_dir, phase='train', cfg=cfg)
        dataset_val = CLEVRTEX(data_dir=args.data_dir, phase='val', cfg=cfg)
        if cfg.WEAK_SUP.SPLIT.RATIO < 1 or use_batch_fusion:
            dataset_train_sub = CLEVRTEX(data_dir=args.data_dir, phase='train', sub=True, cfg=cfg)

    if cfg.DATA.TYPE.lower() == 'ptr':
        dataset_train = PTR(data_dir=args.data_dir, phase='train', cfg=cfg)
        dataset_val = PTR(data_dir=args.data_dir, phase='val', cfg=cfg)
        if cfg.WEAK_SUP.SPLIT.RATIO < 1 or use_batch_fusion:
            dataset_train_sub = PTR(data_dir=args.data_dir, phase='train', sub=True, cfg=cfg)

    elif cfg.DATA.TYPE.lower() == 'movi': 
        dataset_train = MOVi(data_dir=args.data_dir, phase='train', cfg=cfg)
        dataset_val = MOVi(data_dir=args.data_dir, phase='val', cfg=cfg)
        if cfg.WEAK_SUP.SPLIT.RATIO < 1 or use_batch_fusion:
            dataset_train_sub = MOVi(data_dir=args.data_dir, phase='train', sub=True, cfg=cfg)

    print(f"Dataset size: train({len(dataset_train)}), train_sub({len(dataset_train_sub)}), val({len(dataset_val)})")
    
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")

    if use_batch_fusion:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, 
            pin_memory=True, 
            batch_size=cfg.TRAIN.BATCH_SIZE-batch_fusion_ws_num_samples, 
            shuffle=True, 
            num_workers=cfg.DATA.NUM_WORKERS)
        data_loader_train_sub = torch.utils.data.DataLoader(
                dataset_train_sub, 
                pin_memory=True, 
                batch_size=batch_fusion_ws_num_samples, 
                shuffle=True, 
                num_workers=cfg.DATA.NUM_WORKERS
            )
    else:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, 
            pin_memory=True, 
            batch_size=cfg.TRAIN.BATCH_SIZE, 
            shuffle=True, 
            num_workers=cfg.DATA.NUM_WORKERS
        )
        if cfg.WEAK_SUP.SPLIT.RATIO < 1:
            data_loader_train_sub = torch.utils.data.DataLoader(
                dataset_train_sub, 
                pin_memory=True, 
                batch_size=cfg.TRAIN.BATCH_SIZE, 
                shuffle=True, 
                num_workers=cfg.DATA.NUM_WORKERS
            )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, 
        pin_memory=True, 
        batch_size=1, 
        shuffle=False, 
        num_workers=cfg.DATA.NUM_WORKERS
    )

    data_loader_vis = torch.utils.data.DataLoader(
        dataset_val, 
        pin_memory=True, 
        batch_size=args.num_vis, 
        shuffle=False, 
        num_workers=cfg.DATA.NUM_WORKERS
    )

    model = SlotAttentionAutoEncoder(cfg, device=device).to(device)

    criterion = nn.MSELoss()
    params = [{'params': model.parameters()}]
    if cfg.TRAIN.OPTIMIZER.lower() == 'adam':
        optimizer = optim.Adam(params, lr=cfg.TRAIN.BASE_LR)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model = model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model = %s" % str(model))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

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
        main_steps = cfg.TRAIN.EPOCHS * len(data_loader_train)
        sub_steps = 0
        total_steps = main_steps + sub_steps

        print(f"Train Steps: Total({total_steps}) = Main({main_steps}) + Sub({sub_steps})")
        if use_batch_fusion:
            print(f"Use batch fusion: Total Batch({cfg.TRAIN.BATCH_SIZE}) = " +
                  f"Main({cfg.TRAIN.BATCH_SIZE - batch_fusion_ws_num_samples}) " +
                  f"Sub({batch_fusion_ws_num_samples})")

        elapsed_time = 0

    warmup_steps = total_steps * cfg.TRAIN.WARMUP_STEP_RATIO
    decay_steps = total_steps * cfg.TRAIN.DECAY_STEP_RATIO

    for epoch in range(epoch, cfg.TRAIN.EPOCHS):

        data_loader = data_loader_train
    
        start_epoch = time.time()
        model.train()
        total_recon_loss = 0
        total_pos_loss = torch.zeros((cfg.MODEL.SLOT.ITERATIONS,))

        backprop_target_losses = ['recon_loss']
        if use_batch_fusion:
            backprop_target_losses.append('pos_loss')
        
        if use_batch_fusion:
            sub_data_iterator = iter(data_loader_train_sub)
            # data_loader = zip(data_loader_train, data_loader_train_sub)
            len_data_loader = len(data_loader_train)
        else:
            len_data_loader = len(data_loader)

        print(f"Backprop target losses: {backprop_target_losses}")
        for sample in tqdm(data_loader, desc='Epoch {}/{}'.format(epoch+1, cfg.TRAIN.EPOCHS), total=len_data_loader):
            if use_batch_fusion:
                try:
                    sample_sub = next(sub_data_iterator)
                except StopIteration:
                    sub_data_iterator = iter(data_loader_train_sub)
                    sample_sub = next(sub_data_iterator)
                for k in sample.keys():
                    sample[k] = torch.cat([sample_sub[k], sample[k]], dim=0)
                del sample_sub

            total_step += 1

            if total_step < warmup_steps:
                lr = cfg.TRAIN.BASE_LR * (total_step / warmup_steps)
            else:
                lr = cfg.TRAIN.BASE_LR

            lr = lr * (cfg.TRAIN.DECAY_RATE ** (total_step / decay_steps))

            optimizer.param_groups[0]['lr'] = lr

            image = sample['image'].to(device)
            if cfg.WEAK_SUP.TYPE != "" and \
               (use_batch_fusion or cfg.WEAK_SUP.INIT_USING_SUP != ''):
                pos = sample[cfg.WEAK_SUP.TYPE].clone().to(device) # for model prediction
                pos_gt = sample[f"{cfg.WEAK_SUP.TYPE}_gt"].clone() # only for calculating loss
            elif cfg.WEAK_SUP.TYPE != "": 
                pos = None
                pos_gt = sample[f"{cfg.WEAK_SUP.TYPE}_gt"].clone() # only for calculating loss
            elif cfg.WEAK_SUP.INIT_USING_SUP:
                pos = sample[cfg.WEAK_SUP.TYPE].clone().to(device) # for model prediction
                pos_gt = None
            else:
                pos = None
                pos_gt = None

            with torch.cuda.amp.autocast(enabled=use_amp):

                outputs = model(**dict(image=image, pos=pos, train=True))

                loss = criterion(outputs['recon_combined'], image)
                total_recon_loss += loss.item()
                    
                pos_loss = None
                if cfg.POS_PRED.USE_POS_PRED:
                    for iter_idx in range(cfg.MODEL.SLOT.ITERATIONS):
                        pos_pred = outputs['pos_pred'][iter_idx]
                        pos_gt = pos_gt.to(device)

                        if use_batch_fusion:
                            pos_pred_full = pos_pred.clone()
                            # extract pos_pred_for_samples_wo_sup and pos_for_samples_wo_sup 
                            # so that they don't participate in gt matching
                            pos_pred = pos_pred[:batch_fusion_ws_num_samples]
                            pos_gt = pos_gt[:batch_fusion_ws_num_samples]

                        pos_gt_aranged = outputs["pos_gt_aranged"]
                        if pos_gt_aranged is None:
                            # matching gt to pred
                            cost_map = torch.cdist(pos_pred, pos_gt).cpu().detach().numpy() # [B, K, K]
                            match_indexes = np.array([linear_sum_assignment(cost_map[i])[1] for i in range(len(pos_gt))]).reshape(-1) # [B*K,]
                            batch_index = [i // pos_gt.shape[1] for i in range(pos_gt.shape[0] * pos_gt.shape[1])]
                            pos_gt_aranged = pos_gt[range(len(pos_gt))][batch_index, match_indexes].reshape(pos_gt.shape)

                        # zero mask invalid matching
                        # valid_matching_mask = (pos_gt_aranged > -1).float().to(pos_pred.device)
                        valid_matching_mask = (pos_gt_aranged > -1).float().to(device)

                        pos_loss = criterion(pos_pred * valid_matching_mask, pos_gt_aranged.to(device) * valid_matching_mask)
                        total_pos_loss[iter_idx] += pos_loss.item()

                        if use_batch_fusion:
                            pos_loss *= cfg.TRAIN.POS_LOSS_WEIGHT
                            scaler.scale(pos_loss).backward(retain_graph=True)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            del outputs['recons'], outputs['masks'], outputs['slots'], outputs['attn']
            del image, sample, outputs, loss, pos_loss

        train_recon_loss = total_recon_loss / len_data_loader
        train_pos_loss = total_pos_loss / len_data_loader

        end_epoch = time.time()
        elapsed_time += end_epoch - start_epoch

        print(
            "Epoch: {}, Recon Loss: {:.3e}, Pos Loss: {}, Time: {}".format(
                epoch+1, 
                train_recon_loss,
                train_pos_loss,
                datetime.timedelta(seconds=int(elapsed_time))
            )
        )

        if log_writer is not None: 
            print('log_dir: {}\n'.format(log_writer.log_dir))
            log_writer.add_scalar('train_recon_loss', train_recon_loss, epoch+1)
            for iter_idx in range(cfg.MODEL.SLOT.ITERATIONS):
                log_writer.add_scalar(f'train_pos_loss_{iter_idx}', train_pos_loss[iter_idx], epoch+1)
            log_writer.add_scalar('lr', lr, epoch+1)

        # save ckpt 
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),

            'epoch': epoch + 1,
            'epochs': cfg.TRAIN.EPOCHS, 
            'total_step': total_step, 
            'total_steps': total_steps, 
            'warmup_step_ratio': cfg.TRAIN.WARMUP_STEP_RATIO,
            'decay_step_ratio': cfg.TRAIN.DECAY_STEP_RATIO,
            'decay_rate': cfg.TRAIN.DECAY_RATE,
            'elapsed_time': elapsed_time
        }

        prev_checkpoint_list = os.listdir(cfg.OUTPUT.DIR)
        for f in prev_checkpoint_list:
            if ".pth" in f and "00.pth" not in f:
                os.remove(os.path.join(cfg.OUTPUT.DIR, f))

        torch.save(
            checkpoint,
            os.path.join(cfg.OUTPUT.DIR, f'checkpoint-{epoch+1}.pth'),
        )
        torch.save(
            checkpoint,
            os.path.join(cfg.OUTPUT.DIR, f'checkpoint-latest.pth'),
        )

        # evaluation
        if (epoch+1) % cfg.TRAIN.EVAL_INTERVAL == 0: 
            with torch.no_grad(): 

                start_epoch = time.time()
                
                model.eval()
                val_loss = 0
                total_val_recon_loss = 0
                total_val_pos_loss = torch.zeros((cfg.MODEL.SLOT.ITERATIONS,))
                ari_evaluator = ARIEvaluator()
                f_ari_evaluator = ARIEvaluator()
                miou_evaluator = mIoUEvaluator()
                f_miou_evaluator = mIoUEvaluator()

                val_loader_list = [data_loader_val]
                for loader_val in val_loader_list:
                    print('val set!')
                    for sample in tqdm(loader_val, desc='Val {}/{}'.format(epoch+1, cfg.TRAIN.EPOCHS)):
                        image = sample['image'].to(device)
                        if cfg.WEAK_SUP.TYPE != "":
                            pos_gt = sample[cfg.WEAK_SUP.TYPE].clone().detach() # only for calculating loss
                            pos = None
                        else:
                            pos_gt = None
                            pos = None
                        
                        outputs = model(**dict(image=image, pos=pos, train=False))
                        
                        attns = torch.stack(outputs['attns'], dim=1)
                        # `attns`: (B, T, N_heds, N_in, K)

                        recon_combined = outputs['recon_combined']
                        recons = outputs['recons']
                        
                        loss = criterion(recon_combined, image)
                        total_val_recon_loss += loss.item()

                        if cfg.POS_PRED.USE_POS_PRED:
                            for iter_idx in range(cfg.MODEL.SLOT.ITERATIONS):
                                # `pos_pred`: (B, K, 2)
                                # `pos_gt`: (B, K', 2)
                                pos_pred = outputs['pos_pred'][iter_idx]
                                pos_gt = pos_gt.to(device)

                                # cal cost map to match gt to pred
                                cost_map = torch.cdist(pos_pred, pos_gt).cpu().detach().numpy() # [B, K, K]
                                # match gt and pred using linear sum assignment
                                match_indexes = np.array([linear_sum_assignment(cost_map[i])[1] for i in range(len(pos_gt))]).reshape(-1) # [B*K,]
                                batch_index = [i // pos_gt.shape[1] for i in range(pos_gt.shape[0] * pos_gt.shape[1])]
                                pos_gt_aranged = pos_gt[range(len(pos_gt))][batch_index, match_indexes].reshape(pos_gt.shape)
                                # zero mask invalid matching
                                pos_gt_aranged[pos_gt_aranged < -1] = 0.
                                pos_pred[pos_gt_aranged < -1] = 0.

                                pos_loss = criterion(pos_pred, pos_gt_aranged)
                                total_val_pos_loss[iter_idx] += pos_loss.item()

                        masks = outputs['masks']
                        f_ari_evaluator.evaluate(masks, sample['masks'][:, 1:], device)
                        ari_evaluator.evaluate(masks, sample['masks'], device)
                        f_miou_evaluator.evaluate(masks, sample['masks'][:, 1:], device)
                        miou_evaluator.evaluate(masks, sample['masks'], device)

                    val_recon_loss = total_val_recon_loss / len(loader_val)
                    val_pos_loss = total_val_pos_loss / len(loader_val)

                    val_ari = ari_evaluator.get_results()
                    val_f_ari = f_ari_evaluator.get_results()
                    val_miou = miou_evaluator.get_results()
                    val_f_miou = f_miou_evaluator.get_results()

                    end_epoch = time.time()

                    print(
                        "Val: F-ARI: {:.4f}, ARI: {:.4f}, F-mIoU: {:.4f}, mIoU: {:.4f}, Total Loss: {:.3e}, Recon Loss: {:.3e}, Pos Loss: {}, Time: {}".format(
                            val_f_ari,
                            val_ari,
                            val_f_miou,
                            val_miou,
                            val_loss,
                            val_recon_loss, 
                            val_pos_loss,
                            datetime.timedelta(seconds=end_epoch - start_epoch)
                        )
                    )

                    if log_writer is not None: 
                        print(f'log_dir: {log_writer.log_dir}\n')
                        log_writer.add_scalar(f'val_loss', val_loss, epoch+1)
                        log_writer.add_scalar(f'val_recon_loss', val_recon_loss, epoch+1)
                        for iter_idx in range(cfg.MODEL.SLOT.ITERATIONS):
                            log_writer.add_scalar(f'val_pos_loss_{iter_idx+1}', val_pos_loss[iter_idx], epoch+1)
                        log_writer.add_scalar(f'val_ari', val_ari, epoch+1)
                        log_writer.add_scalar(f'val_f_ari', val_f_ari, epoch+1)
                        log_writer.add_scalar(f'val_miou', val_miou, epoch+1)
                        log_writer.add_scalar(f'val_f_miou', val_f_miou, epoch+1)
                        
                        sample = next(iter(data_loader_vis))
                        image = sample['image'].to(device)
                        if cfg.WEAK_SUP.TYPE != "":
                            pos_gt = sample[cfg.WEAK_SUP.TYPE].clone().detach() # only for calculating loss
                            pos = None
                        else:
                            pos_gt = None
                            pos = None
                        outputs = model(**dict(image=image, pos=pos, train=False))
                        attns = torch.stack(outputs['attns'], dim=1)

                        if cfg.POS_PRED.USE_POS_PRED:
                            pos_pred = torch.stack(outputs['pos_pred'], dim=1)
                            # `pos_pred_origin`: (B, L, K, 2)
                        else: 
                            pos_pred = None 
                        
                        log_img = visualize(image=image, 
                                            recon_combined=outputs['recon_combined'],
                                            recons=outputs['recons'], 
                                            masks=outputs['masks'], 
                                            attns=attns, 
                                            pos_pred=pos_pred, 
                                            num_vis=args.num_vis)
                        log_img = vutils.make_grid(log_img, nrow=1, pad_value=0)
                        log_writer.add_image(f'val_visualization/epoch={epoch+1:04}', log_img)
                    
                del outputs, masks, attns, image, recons, recon_combined, log_img   
                del ari_evaluator, f_ari_evaluator, miou_evaluator, f_miou_evaluator


    if log_writer is not None:
        log_writer.close()

if __name__ == "__main__":
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    parser = argparse.ArgumentParser('Slot Attention training script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args=args)

