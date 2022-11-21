import time 
import datetime
import argparse
from tqdm import tqdm
from pathlib import Path 

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from datasets import *
from datasets_sp import *
from model import *
from utils.config import *
from utils import misc

from utils.evaluator import *

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config_file', default='configs/config.yaml', type=str)
    parser.add_argument('--data_dir', default='/workspace/dataset/clevr_with_masks/CLEVR6', type=str)
    parser.add_argument('--batch_size', default=0, type=int, help='Desired batch_size: 64 x num_gpus')
    parser.add_argument('--lr', default=0, type=float)
    parser.add_argument('--eval_interval', default=0, type=int)
    parser.add_argument('--num_workers', default=-1, type=int)
    parser.add_argument('--output_dir_suffix', default='', type=str)
    parser.add_argument('--model_ckpt', default='', type=str)
    parser.add_argument('--resume_ckpt', default='', type=str)
    parser.add_argument('--epochs', default=0, type=int)
    parser.add_argument('--use_pos_sup', action='store_true')

    return parser

def main(rank, world_size, args):
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
    if cfg.OUTPUT.DIR[-1] == '/':
        cfg.OUTPUT.DIR = f"{cfg.OUTPUT.DIR[:-1]}_{args.output_dir_suffix}"
    else:
        cfg.OUTPUT.DIR = f"{cfg.OUTPUT.DIR}_{args.output_dir_suffix}"

    if cfg.OUTPUT.DIR is not None: 
        Path(cfg.OUTPUT.DIR).mkdir(parents=True, exist_ok=True)
        log_writer = SummaryWriter(log_dir=cfg.OUTPUT.DIR)
    else: 
        log_writer = None

    if cfg.DATA.TYPE in ['ptr', 'PTR']:
        dataset_train = PTR(data_dir=args.data_dir, phase='train', cfg=cfg)
        dataset_val = PTR(data_dir=args.data_dir, phase='val', cfg=cfg)

    elif cfg.DATA.TYPE in ['clevr', 'CLEVR']:
        dataset_train = CLEVR(data_dir=args.data_dir, phase='train', cfg=cfg)
        dataset_val = CLEVR(data_dir=args.data_dir, phase='val', cfg=cfg)

    elif cfg.DATA.TYPE.lower() == 'clevrtex':
        dataset_train = CLEVRTEX(data_dir=args.data_dir, phase='train', cfg=cfg)
        dataset_val = CLEVRTEX(data_dir=args.data_dir, phase='val', cfg=cfg)

    elif cfg.DATA.TYPE in ['mdg', 'MDG']: 
        dataset_train = MultiDspritesGray(data_dir=args.data_dir, phase='train', cfg=cfg)
        dataset_val = MultiDspritesGray(data_dir=args.data_dir, phase='val', cfg=cfg)
    
    elif cfg.DATA.TYPE in ['tet', 'TET']: 
        dataset_train = Tetrominoes(data_dir=args.data_dir, phase='train', cfg=cfg)
        dataset_val = Tetrominoes(data_dir=args.data_dir, phase='val', cfg=cfg)

    elif cfg.DATA.TYPE.lower() == 'movi': 
        dataset_train = MOVi(data_dir=args.data_dir, phase='train', cfg=cfg)
        dataset_val = MOVi(data_dir=args.data_dir, phase='val', cfg=cfg)

    use_amp = True
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        pin_memory=True, 
        batch_size=cfg.TRAIN.BATCH_SIZE, 
        shuffle=True, 
        num_workers=cfg.DATA.NUM_WORKERS
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, 
        pin_memory=True, 
        batch_size=cfg.TRAIN.BATCH_SIZE, 
        shuffle=False, 
        num_workers=cfg.DATA.NUM_WORKERS
    )

    model = SlotAttentionClassifier(cfg, device=device).to(device)
    model = nn.DataParallel(model).to(device)
    params = [{'params': model.module.parameters()}]
    n_parameters = sum(p.numel() for p in model.module.parameters() if p.requires_grad)

    if cfg.SET_PREDICTION.LOSS == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif cfg.SET_PREDICTION.LOSS == 'huber':
        criterion = nn.HuberLoss()
    
    # optimizer = optim.Adam(params, lr=cfg.TRAIN.BASE_LR)
    optimizer = optim.AdamW(params, lr=cfg.TRAIN.BASE_LR)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print("Device = %s" % device)
    print("Model = %s" % str(model))
    print('number of params (M) = %.2f' % (n_parameters / 1.e6))

    if args.resume_ckpt != '': 
        assert os.path.exists(args.resume_ckpt), "Wrong checkpoint!"

        checkpoint = torch.load(args.resume_ckpt, map_location=device)
        model.module.load_state_dict(checkpoint['model_state_dict'])
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
        assert os.path.exists(args.model_ckpt), "Wrong checkpoint!"
        
        # TODO: load checkpoint & freeze parameters 
        checkpoint = torch.load(args.model_ckpt, map_location=device)
        model.module.load_state_dict(checkpoint['model_state_dict'], strict=False)

        epoch = 0
        total_step = 0
        main_steps = cfg.TRAIN.EPOCHS * len(data_loader_train)
        sub_steps = 0
        total_steps = main_steps + sub_steps
    
    # Freezeing parameters 
    for param in model.module.encoder_cnn.parameters():
        param.requires_grad = False 
    for param in model.module.slot_attention.parameters():
        param.requires_grad = False 

    warmup_steps = total_steps * cfg.TRAIN.WARMUP_STEP_RATIO
    decay_steps = total_steps * cfg.TRAIN.DECAY_STEP_RATIO

    for epoch in range(epoch, cfg.TRAIN.EPOCHS):
        data_loader = data_loader_train
        len_data_loader = len(data_loader)
        train_iterator = tqdm(data_loader, desc='Epoch {}/{}'.format(epoch+1, cfg.TRAIN.EPOCHS), total=len_data_loader)
        
        train_loss = 0

        model.train()
        elapsed_time = 0
        start_epoch = time.time()
        for sample in train_iterator:
            total_step += 1
            lr = cfg.TRAIN.BASE_LR
            lr = lr * (cfg.TRAIN.DECAY_RATE ** (total_step / decay_steps))
            optimizer.param_groups[0]['lr'] = lr

            if cfg.POS_PRED.USE_POS_PRED and cfg.POS_PRED.USE_GT and cfg.WEAK_SUP.TYPE != "":
                pos = sample[cfg.WEAK_SUP.TYPE]
            else:
                pos = None

            image = sample['image'].to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(**dict(image=image, pos=pos, train=True))

                predictions = outputs['predictions'] # (B, K, num_attr)
                real_obj_pred = predictions[..., -1:] # (B, K, 1)
                predictions = predictions[..., :-1] 

                pred_mask = (real_obj_pred > 1e-4).float().detach() # (B, K, 1)
                predictions = predictions * pred_mask + torch.full_like(predictions, -1).to(device) * (1 - pred_mask)

                assert len(cfg.SET_PREDICTION.ATTRIBUTES) > 0
                for attr_i, attr in enumerate(cfg.SET_PREDICTION.ATTRIBUTES):
                    if attr_i == 0:
                        attr_gt = sample[attr]
                    else:
                        attr_gt = torch.cat([attr_gt, sample[attr]], dim=2) 
                if not (cfg.POS_PRED.USE_POS_PRED and cfg.SET_PREDICTION.USE_SEPARATE_POS):
                    attr_gt = torch.cat([sample['pixel_coords'], attr_gt], dim=2) # make coords range -0.5~0.5
                attr_gt = attr_gt.float().to(device) # (B, K, num_attr)
                real_obj_gt = sample['real_obj'].to(device)

                pos_gt = sample['pixel_coords'].to(device) # (B, K, 2)
                if cfg.POS_PRED.USE_POS_PRED and cfg.SET_PREDICTION.USE_SEPARATE_POS:
                    # from the pos_pred list, use the last prediction
                    pos_pred = outputs['pos_pred'][-1] # (B, K, 2), range -0.5~0.5
                else:
                    predictions[..., :2] -= 0.5 # to make range -0.5~0.5
                    pos_pred = predictions[..., :2]
                
                # cost_map = torch.cdist(pos_pred, pos_gt).detach().cpu().numpy() # [B, K, K]
                if cfg.POS_PRED.USE_POS_PRED and cfg.SET_PREDICTION.USE_SEPARATE_POS:
                    cost_map = torch.cdist(torch.cat([pos_pred, predictions], dim=2), torch.cat([pos_gt, attr_gt], dim=2)).detach().cpu().numpy() # [B, K, K]
                else:
                    cost_map = torch.cdist(predictions, attr_gt).detach().cpu().numpy() # [B, K, K]
                match_indexes = np.array([linear_sum_assignment(cost_map[i])[1] for i in range(len(attr_gt))]).reshape(-1) # [B*K,]
                batch_index = [i // attr_gt.shape[1] for i in range(attr_gt.shape[0] * attr_gt.shape[1])]
                # pos_gt_aranged = pos_gt[range(len(pos_gt))][batch_index, match_indexes].reshape(pos_gt.shape)
                attr_gt_aranged = attr_gt[range(len(attr_gt))][batch_index, match_indexes].reshape(attr_gt.shape)

                # valid_matching_mask = (pos_gt_aranged[..., :1] > -1).float().to(device) # value for no_obj in pos_gt == -1.5
                valid_matching_mask = pred_mask * (real_obj_gt > 1e-4).float().detach()

                # calculate loss with valid_matching_mask
                # except for the real_obj attr in predictions
                loss = criterion(predictions * valid_matching_mask, attr_gt_aranged * valid_matching_mask)
                train_loss += loss.item()

                scaler.scale(loss).backward()
                # loss.backward()
                scaler.step(optimizer)
                scaler.update()
                # optimizer.step()
                optimizer.zero_grad()

        end_epoch = time.time()
        elapsed_time += end_epoch - start_epoch
        train_loss = train_loss / len_data_loader

        print(
            "Epoch: {}, Total Loss: {:.3e}, Time: {}".format(
                epoch+1, 
                train_loss, 
                datetime.timedelta(seconds=int(elapsed_time))
            )
        )

        if log_writer is not None: 
            print('log_dir: {}\n'.format(log_writer.log_dir))
            log_writer.add_scalar('train_loss', train_loss, epoch+1)
            log_writer.add_scalar('lr', lr, epoch+1)

        if not (epoch + 1) % 25 or (epoch + 1) == cfg.TRAIN.EPOCHS:
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
            }

            torch.save(
                checkpoint,
                os.path.join(cfg.OUTPUT.DIR, f'checkpoint-{epoch+1}.pth'),
            )

        if (epoch+1) % cfg.TRAIN.EVAL_INTERVAL == 0:
            start_epoch = time.time()
            with torch.no_grad(): 
                model.eval()
                val_loss = 0
                val_ap_dict = dict([(dist_thrs, 0) for dist_thrs in cfg.SET_PREDICTION.DISTANCE_THRESHOLDS])
                len_data_loader = len(data_loader_val)
                val_iterator = tqdm(data_loader_val, desc='Val{}/{}'.format(epoch+1, cfg.TRAIN.EPOCHS))
                for sample in val_iterator:
                    image = sample['image'].to(device)

                    if args.use_pos_sup and cfg.WEAK_SUP.TYPE != "":
                        pos = sample[cfg.WEAK_SUP.TYPE]
                    else:
                        pos = None

                    outputs = model(**dict(image=image, pos=pos, train=False))
                    predictions = outputs['predictions']
                    pred_mask = (predictions[..., -1:] > 1e-4).float().detach() # (B, K, 1)
                    
                    assert len(cfg.SET_PREDICTION.ATTRIBUTES) > 0
                    for attr_i, attr in enumerate(cfg.SET_PREDICTION.ATTRIBUTES + ['real_obj']):
                        if attr_i == 0:
                            attr_gt = sample[attr]
                        else:
                            attr_gt = torch.cat([attr_gt, sample[attr]], dim=2) 
                    if not (cfg.POS_PRED.USE_POS_PRED and cfg.SET_PREDICTION.USE_SEPARATE_POS):
                        attr_gt = torch.cat([sample['pixel_coords'], attr_gt], dim=2) # coords range -0.5~0.5
                    attr_gt = attr_gt.float().to(device) # (B, K, num_attr)
                    real_obj_gt = sample['real_obj'].to(device)

                    pos_gt = sample['pixel_coords'].to(device) # (B, K, 2)
                    if cfg.POS_PRED.USE_POS_PRED and cfg.SET_PREDICTION.USE_SEPARATE_POS:
                        pos_pred = outputs['pos_pred'][-1] # (B, K, 2), range -0.5~0.5
                    else:
                        predictions[..., :2] -= 0.5 # make coords range -0.5~0.5
                        pos_pred = predictions[..., :2]
                    
                    # TODO: do we need calculating loss for val?
                    if cfg.POS_PRED.USE_POS_PRED and cfg.SET_PREDICTION.USE_SEPARATE_POS:
                        cost_map = torch.cdist(torch.cat([pos_pred, predictions], dim=2), torch.cat([pos_gt, attr_gt], dim=2)).detach().cpu().numpy() # [B, K, K]
                    else:
                        cost_map = torch.cdist(predictions, attr_gt).detach().cpu().numpy() # [B, K, K]
                    match_indexes = np.array([linear_sum_assignment(cost_map[i])[1] for i in range(len(pos_gt))]).reshape(-1) # [B*K,]
                    batch_index = [i // pos_gt.shape[1] for i in range(pos_gt.shape[0] * pos_gt.shape[1])]
                    pos_gt_aranged = pos_gt[range(len(pos_gt))][batch_index, match_indexes].reshape(pos_gt.shape)
                    attr_gt_aranged = attr_gt[range(len(pos_gt))][batch_index, match_indexes].reshape(attr_gt.shape)
                    valid_matching_mask = pred_mask * (real_obj_gt > 1e-4).float().detach()

                    # calculate loss with valid_matching_mask
                    # except for the real_obj attr in predictions
                    loss = criterion(predictions[..., :-1] * valid_matching_mask, attr_gt_aranged[..., :-1] * valid_matching_mask)
                    val_loss += loss.item()

                    # calculate AP for given distatnce thresholds
                    for dist_thrs in cfg.SET_PREDICTION.DISTANCE_THRESHOLDS:
                        if cfg.POS_PRED.USE_POS_PRED and cfg.SET_PREDICTION.USE_SEPARATE_POS:
                            if cfg.DATA.TYPE.lower() in ['clevr']:
                                ap = average_precision_clevr(predictions, attr_gt, pos_pred, pos_gt, dist_thrs)
                            elif cfg.DATA.TYPE.lower() in ['ptr', 'movi', 'clevrtex']:
                                ap = average_precision_ptr(predictions, attr_gt, pos_pred, pos_gt, dist_thrs)
                        else:
                            if cfg.DATA.TYPE.lower() in ['clevr']:
                                ap = average_precision_clevr(predictions[..., 2:], attr_gt[..., 2:], pos_pred, pos_gt, dist_thrs)
                            elif cfg.DATA.TYPE.lower() in ['ptr', 'movi', 'clevrtex']:
                                ap = average_precision_ptr(predictions[..., 2:], attr_gt[..., 2:], pos_pred, pos_gt, dist_thrs)
                
                        val_ap_dict[dist_thrs] += ap

            end_epoch = time.time()
            val_loss = val_loss / len_data_loader
            for dist_thrs in cfg.SET_PREDICTION.DISTANCE_THRESHOLDS:
                val_ap_dict[dist_thrs] /= len_data_loader
            
            print(
                "Val Loss: {:.3e}, Val AP: {}, Time: {}".format(
                    val_loss,
                    val_ap_dict,
                    datetime.timedelta(seconds=end_epoch - start_epoch)
                )
            )

            if log_writer is not None: 
                print(f'log_dir: {log_writer.log_dir}\n')
                log_writer.add_scalar('val_loss', val_loss, epoch+1)
                for dist_thrs, ap_value in val_ap_dict.items():
                    log_writer.add_scalar(f'val_ap_for_{dist_thrs}', ap_value, epoch+1)

    if log_writer is not None:
        log_writer.close()

if __name__ == "__main__":
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    parser = argparse.ArgumentParser('Set Prediction training script', parents=[get_args_parser()])
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    main(rank=None, world_size=world_size, args=args)