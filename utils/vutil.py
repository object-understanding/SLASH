import torch 
import torch.nn.functional as F 
import numpy as np 

def visualize(image, recon_combined, recons, masks, attns=None, pos_pred=None, pos_pred_loc=None, pos_existence_mask=None, num_vis=4, colored_box=True):
    '''
        `image`: [B, 3, H, W]
        `recon_combined`: [B, 3, H, W]
        `recons`: [B, K, H, W, C]
        `masks`: [B, K, H, W, 1]
        `attns`: (B, T, N_heads, N_in, K)
        `pos_pred`: (B, L, K, 2)
        `pos_existence_mask`: (B, L, K, 1)

    '''

    N_samples = num_vis

    if (pos_pred_loc is not None) and (0 in pos_pred_loc): 
        pos_pred_loc = pos_pred_loc[1:]

    image = image[:N_samples]
    recon_combined = recon_combined[:N_samples]
    recons = recons[:N_samples]
    masks = masks[:N_samples]
    attns = attns[:N_samples] 

    _, _, H, W = image.shape
    _, T, _, _, K = attns.shape
    
    N_samples = min(N_samples, image.shape[0])
    
    # get binarized masks 
    mask_max_idxs = torch.argmax(masks.squeeze(-1), dim=1)
    # `mask_max_idxs`: (N_samples, H, W)
    binary_masks = torch.zeros_like(masks.squeeze(-1))
    binary_masks[torch.arange(N_samples)[:, None, None], mask_max_idxs, torch.arange(H)[None, :, None], torch.arange(W)[None, None, :]] = 1.0
    binary_masks = binary_masks.unsqueeze(-1)
    # `binary_masks`: (N_samples, K, H, W, 1)

    pad = (0, 0, 2, 2, 2, 2)

    # set colors
    mark_colors = torch.tensor(np.array([[139,69,19], [255, 0, 0], [0, 255, 0], [0, 255, 255], [255, 0, 255], [0, 0, 255]]), dtype=torch.float32) / 255.0 
    
    slot_colors = torch.tensor(np.array([[255, 0, 0], [255, 127, 0], [255, 255, 0], [0, 255, 0], 
                                         [0, 0, 255], [75, 0, 130], [148, 0, 211], 
                                         [255, 0, 0], [255, 127, 0], [255, 255, 0], [0, 255, 0], 
                                         [0, 0, 255], [75, 0, 130], [148, 0, 211],
                                         [0, 0, 0], [255, 255, 255]]), dtype=torch.float32) / 255.0 

    # handle the multi-head attention
    attns = torch.mean(attns[:N_samples], dim=2).permute(0, 1, 3, 2).view(N_samples, T, K, H, W).unsqueeze(-1)
    # `attns`: (N_samples, T, K, H, W, 1)

    # concat alpha mask to the end of `attns`
    attns = torch.cat([attns, masks.unsqueeze(1), binary_masks.unsqueeze(1)], dim=1) 
    # `attns`: (N_samples, T+2, K, H, W, 1)

    # reshape tensors
    attns = attns.reshape(N_samples*(T+2), K, H, W, 1)
    # `attns`: (N_samples*(T+2), K, H, W, 1)

    image = torch.repeat_interleave(image, T+2, dim=0)
    # `image`: (N_samples*(T+2), 3, H, W)

    # old: recon_combined = torch.repeat_interleave(recon_combined, T+1, dim=0)
    recon_combined_template = torch.zeros_like(image) 
    recon_combined_template[torch.arange(T, N_samples*(T+2), T+2)] = recon_combined
    recon_combined_template[torch.arange(T+1, N_samples*(T+2), T+2)] = recon_combined
    recon_combined = recon_combined_template
    # `recon_combined`: (N_samples*(T+2), 3, H, W)

    recons = torch.repeat_interleave(recons, T+2, dim=0)
    # `recons`: (N_samples*(T+2), K, H, W, 3)

    # draw boundary box for the original image
    image = torch.einsum('nchw->nhwc', image)
    image = F.pad(image, pad=pad, mode='constant', value=1.0)
    if colored_box:
        image[:, :2, :] = image[:, -2:, :] = image[:, :, :2] = image[:, :, -2:] = slot_colors[-1]
    image = F.pad(image, pad=pad, mode='constant', value=1.0)
    # image: [N_samples*(T+2), H, W, C]

    # draw boundary box for the reconstructed image
    recon_combined = torch.einsum('nchw->nhwc', recon_combined)
    recon_combined = F.pad(recon_combined, pad=pad, mode='constant', value=1.0)
    if colored_box:
        recon_combined[:, :2, :] = recon_combined[:, -2:, :] = recon_combined[:, :, :2] = recon_combined[:, :, -2:] = slot_colors[-2]
    recon_combined = F.pad(recon_combined, pad=pad, mode='constant', value=1.0)
    # recon_combined: [N_samples*(T+2), H, W, C]

    # get idxs to draw mark for `pos_pred`
    if pos_pred is not None:
        pos_pred = pos_pred[:N_samples]
        _, L_pred, _, _ = pos_pred.shape

        # denormalize `pos_pred`
        denormed_pos_pred = torch.stack([ W * (pos_pred[..., 0] + 0.5), H * (pos_pred[..., 1] + 0.5)], dim=-1)
        # `denormed_pos_pred`: (N_samples, L_pred, K, 2)

        # get the position to draw mark
        mark_start = denormed_pos_pred - 2
        mark_start = torch.where(mark_start > 0, mark_start, torch.zeros_like(mark_start, dtype=torch.float32, device=mark_start.device))
        mark_start = mark_start.long()
        # `mark_start`: (N_samples, L_pred, K, 2)

        mark_end = denormed_pos_pred + 2
        mark_end[..., 0] = torch.where(mark_end[..., 0] < W, mark_end[..., 0], torch.zeros_like(mark_end[..., 0], dtype=torch.float32, device=mark_end.device))
        mark_end[..., 1] = torch.where(mark_end[..., 1] < H, mark_end[..., 1], torch.zeros_like(mark_end[..., 1], dtype=torch.float32, device=mark_end.device))
        mark_end = mark_end.long()
        # `mark_end`: (N_samples, L_pred, K, 2)

    if pos_existence_mask is not None: 
        pos_existence_mask = pos_existence_mask[:N_samples]
        # `pos_existence_mask`: (N_samples, K, 1)

    log_img = torch.cat([image, recon_combined], dim=2)
    for k in range(K):

        # get vis. of attention maps based on the original image
        picture = image[:, 4:-4, 4:-4, :] * attns[:, k, :, :, :] + (1 - attns[:, k, :, :, :])
        # `picture`: (N_samples*(T+2), H, W, 3)

        # overwrite vis. of alpha mask with the recon. image
        alpha_idxs = torch.arange(T, image.shape[0], T+2)
        picture[alpha_idxs] = recons[alpha_idxs, k, :, :, :] * attns[alpha_idxs, k, :, :, :] + (1 - attns[alpha_idxs, k, :, :, :])

        binary_idxs = torch.arange(T+1, image.shape[0], T+2)
        # picture[binary_idxs] = recons[binary_idxs, k, :, :, :] * binary_masks[:, k, :, :, :] + (1 - binary_masks[:, k, :, :, :])
        # picture[binary_idxs] = torch.ones_like(recons[binary_idxs, k, :, :, :]) * binary_masks[:, k, :, :, :]
        picture[binary_idxs] = binary_masks[:, k, :, :, :]

        if pos_pred is not None: 
            for n in range(N_samples): 
                for enum_idx, l in enumerate(pos_pred_loc):

                    # draw mark for each iteration on the corresponding attn map 
                    picture[n*(T+2)+(l-1), mark_start[n, enum_idx, k, 1]:mark_end[n, enum_idx, k, 1], mark_start[n, enum_idx, k, 0]:mark_end[n, enum_idx, k, 0], :] = mark_colors[l]

                    # draw mark for all iterations on the alpha mask 
                    picture[n*(T+2)+T, mark_start[n, enum_idx, k, 1]:mark_end[n, enum_idx, k, 1], mark_start[n, enum_idx, k, 0]:mark_end[n, enum_idx, k, 0], :] = mark_colors[l]

        # draw boundary box for slots
        picture = F.pad(picture, pad=pad, mode='constant', value=1.0)
        if colored_box:
            picture[:, :2, :] = picture[:, -2:, :] = picture[:, :, :2] = picture[:, :, -2:] = slot_colors[k]
        picture = F.pad(picture, pad=pad, mode='constant', value=1.0)
        
        if pos_existence_mask is not None: 
            for n in range(N_samples): 
                for enum_idx, l in enumerate(pos_pred_loc):
                    if pos_existence_mask[n, enum_idx, k, 0] == 0:
                        picture[n*(T+2)+(l-1), :4, :] = picture[n*(T+2)+(l-1), -4:, :] = picture[n*(T+2)+(l-1), :, :4] = picture[n*(T+2)+(l-1), :, -4:] = 1.

        log_img = torch.cat([log_img, picture], dim=2)

    log_img = log_img.permute(0, 3, 1, 2)
    return log_img

def visualize_attns(image, recon_combined, recons, pred_masks, gt_masks=None, attns=None, attns_origin=None, 
                    pos_pred=None, pos_pred_loc=None, pos_existence_mask=None, colored_box=True):
    '''
        `image`: [B, 3, H, W]
        `recon_combined`: [B, 3, H, W]
        `recons`: [B, K, H, W, C]
        `pred_masks`: [B, K, H, W, 1]
        `gt_masks`: [B, K, H, W, 1]
        `attns`: (B, T, N_heads, N_in, K)
        `attns_origin`: (B, T, N_heads, N_in, K)
        `pos_pred`: (B, L, K, 2)
        `pos_existence_mask`: (B, L, K, 1)

    '''

    image = image[:1]
    recon_combined = recon_combined[:1]
    recons = recons[:1]
    pred_masks = pred_masks[:1]
    gt_masks = gt_masks[:1]
    attns = attns[:1] 

    _, _, H, W = image.shape
    _, T, _, _, K = attns.shape

    
    # get binarized masks 
    pred_mask_max_idxs = torch.argmax(pred_masks.squeeze(-1), dim=1)
    # `mask_max_idxs`: (1, H, W)
    pred_seg_masks = torch.zeros_like(pred_masks.squeeze(-1))
    pred_seg_masks[torch.arange(1)[:, None, None], pred_mask_max_idxs, torch.arange(H)[None, :, None], torch.arange(W)[None, None, :]] = 1.0
    pred_seg_masks = pred_seg_masks.unsqueeze(-1)
    # `binary_masks`: (1, K, H, W, 1)

    pad = (0, 0, 2, 2, 2, 2)

    # set colors
    mark_colors = torch.tensor(np.array([[139,69,19], [255, 0, 0], [0, 255, 0], [0, 255, 255], [255, 0, 255], [0, 0, 255]]), dtype=torch.float32) / 255.0 
    
    slot_colors = torch.tensor(np.array([[255, 0, 0], [255, 127, 0], [255, 255, 0], [0, 255, 0], 
                                         [0, 0, 255], [75, 0, 130], [148, 0, 211], 
                                         [0, 255, 255], [153, 255, 153], [255, 153, 204], [102, 0, 51],
                                         [255, 255, 255], [0, 0, 0], [128, 128, 128]]), dtype=torch.float32) / 255.0 

    # handle the multi-head attention
    attns = torch.mean(attns, dim=2).permute(0, 1, 3, 2).view(1, T, K, H, W).unsqueeze(-1)
    # `attns`: (1, T, K, H, W, 1)
    if attns_origin is not None: 
        attns_origin = attns_origin[:1]
        attns_origin = torch.mean(attns_origin, dim=2).permute(0, 1, 3, 2).view(1, T, K, H, W).unsqueeze(-1)
        # `attns_origin`: (1, T, K, H, W, 1)

        attns_template = torch.zeros(1, T*2, K, H, W, 1, dtype=torch.float32, device=attns.device) 
        attns_template[:, torch.arange(0, T*2, 2)] = attns_origin
        attns_template[:, torch.arange(1, T*2, 2)] = attns
        attns = attns_template
        # `attns`: (1, T*2, K, H, W, 1)

    # reshape tensors
    if attns_origin is None: 
        attns = attns.reshape(T, K, H, W, 1)
        # `attns`: (T, K, H, W, 1)
    else: 
        attns = attns.reshape(T*2, K, H, W, 1)
        # `attns`: (T*2, K, H, W, 1)
    
    attns = torch.cat([attns, pred_masks, pred_seg_masks], dim=0) 
    N_row = attns.shape[0]
    # `attns`: (N_row, K, H, W, 1)
    
    image = torch.einsum('nchw->nhwc', image)
    # image = torch.repeat_interleave(image, (T*2) if attns_origin is None else (T*2)*2, dim=0) # to vis. colored recon for attn_origin
    if attns_origin is None: 
        image_template = torch.ones((T+2, H, W, 3), dtype=image.dtype, device=image.device)
        image_template[T] = image
        image_template[T+1] = 0 # to contain seg. mask by adding values
    else: 
        image_template = torch.ones((T*2+2, H, W, 3), dtype=image.dtype, device=image.device)
        image_template[T*2] = image
        image_template[T*2+1] = 0 # to contain seg. mask by adding values
    
    # draw boundary box for the original image
    image_template = F.pad(image_template, pad=pad, mode='constant', value=1.0)
    # if colored_box:
    #     image_template[:, :2, :] = image_template[:, -2:, :] = image_template[:, :, :2] = image_template[:, :, -2:] = slot_colors[-1]
    image_template = F.pad(image_template, pad=pad, mode='constant', value=1.0)
    # `image`: [(T*2)*2, H, W, C]


    recon_combined = torch.einsum('nchw->nhwc', recon_combined)
    if attns_origin is None: 
        recon_comb_template = torch.ones((T+2, H, W, 3), dtype=recon_combined.dtype, device=recon_combined.device)
        recon_comb_template[T] = recon_combined
        recon_comb_template[T+1] = 0 # to contain seg. mask by adding values
    else: 
        recon_comb_template = torch.ones((T*2+2, H, W, 3), dtype=recon_combined.dtype, device=recon_combined.device)
        recon_comb_template[T*2] = recon_combined
        recon_comb_template[T*2+1] = 0 # to contain seg. mask by adding values

    # draw boundary box for the reconstructed image
    recon_comb_template = F.pad(recon_comb_template, pad=pad, mode='constant', value=1.0)
    # if colored_box:
    #     recon_combined[:, :2, :] = recon_combined[:, -2:, :] = recon_combined[:, :, :2] = recon_combined[:, :, -2:] = slot_colors[-2]
    recon_comb_template = F.pad(recon_comb_template, pad=pad, mode='constant', value=1.0)

    # get idxs to draw mark for `pos_pred`
    if pos_pred is not None:
        pos_pred = pos_pred[:1]
        _, L_pred, _, _ = pos_pred.shape

        # denormalize `pos_pred`
        denormed_pos_pred = torch.stack([ W * (pos_pred[..., 0] + 0.5), H * (pos_pred[..., 1] + 0.5)], dim=-1)
        # `denormed_pos_pred`: (N_samples, L_pred, K, 2)

        # get the position to draw mark
        mark_start = denormed_pos_pred - 3
        mark_start = torch.where(mark_start > 0, mark_start, torch.zeros_like(mark_start, dtype=torch.float32, device=mark_start.device))
        mark_start = mark_start.long()
        # `mark_start`: (N_samples, L_pred, K, 2)

        mark_end = denormed_pos_pred + 3
        mark_end[..., 0] = torch.where(mark_end[..., 0] < W, mark_end[..., 0], torch.zeros_like(mark_end[..., 0], dtype=torch.float32, device=mark_end.device))
        mark_end[..., 1] = torch.where(mark_end[..., 1] < H, mark_end[..., 1], torch.zeros_like(mark_end[..., 1], dtype=torch.float32, device=mark_end.device))
        mark_end = mark_end.long()
        # `mark_end`: (N_samples, L_pred, K, 2)

    if pos_existence_mask is not None: 
        pos_existence_mask = pos_existence_mask[:1]
        # `pos_existence_mask`: (N_samples, K, 1)

    # log_img = torch.cat([image_template, recon_comb_template], dim=2)
    for k in range(K):

        # # get vis. of attention maps based on the original image
        # if attns_origin is not None:
        #     picture[:T*2] = template * attns[:, k, :, :, :]
        #     picture[T*2:] = image[:T*2, 4:-4, 4:-4, :] * attns[:, k, :, :, :] + (1 - attns[:, k, :, :, :])
        #     # `picture`: (N_samples*(T*2)*2, H, W, 3)
        picture = torch.ones_like(image) * attns[:, k, :, :, :]

        # overwrite vis. of alpha mask with the recon. image
        if attns_origin is None: 
            alpha_idxs = torch.tensor(T, dtype=torch.long, device=recons.device)
        else: 
            alpha_idxs = torch.tensor(T*2, dtype=torch.long, device=recons.device)
        picture[alpha_idxs] = recons[:, k, :, :, :] * attns[alpha_idxs, k, :, :, :] + (1 - attns[alpha_idxs, k, :, :, :])

        if attns_origin is None: 
            binary_idxs = torch.tensor(T+1, dtype=torch.long, device=recons.device)
        else: 
            binary_idxs = torch.tensor(T*2+1, dtype=torch.long, device=recons.device)
        picture[binary_idxs] = pred_seg_masks[:, k, :, :, :]
        recon_comb_template[T*2+1, 4:-4, 4:-4, :] += pred_seg_masks[0, k, :, :, :] * slot_colors[k].to(pred_seg_masks.device)
        image_template[T*2+1, 4:-4, 4:-4, :] += gt_masks[0, k, :, :, :] * slot_colors[k-1].to(pred_seg_masks.device) # `k-1` -> to give white color to background

        if pos_pred is not None: 
            for enum_idx, l in enumerate(pos_pred_loc):

                # draw mark for each iteration on the corresponding attn map 
                picture[2*l-1, mark_start[0, enum_idx, k, 1]:mark_end[0, enum_idx, k, 1], mark_start[0, enum_idx, k, 0]:mark_end[0, enum_idx, k, 0], :] = mark_colors[l]

                # draw mark for all iterations on the alpha mask 
                # picture[(T+2)+T, mark_start[0, enum_idx, k, 1]:mark_end[0, enum_idx, k, 1], mark_start[0, enum_idx, k, 0]:mark_end[0, enum_idx, k, 0], :] = mark_colors[l]

        # draw boundary box for slots
        picture = F.pad(picture, pad=pad, mode='constant', value=1.0)
        if colored_box:
            picture[:, :2, :] = picture[:, -2:, :] = picture[:, :, :2] = picture[:, :, -2:] = slot_colors[k]
        picture = F.pad(picture, pad=pad, mode='constant', value=1.0)

        if k == 0:
            log_img = torch.cat([picture], dim=2)
        else: 
            log_img = torch.cat([log_img, picture], dim=2)
    
    log_img = torch.cat([image_template, recon_comb_template, log_img], dim=2)

    log_img = log_img.permute(0, 3, 1, 2)
    return log_img



def visualize_simple(image, recon_combined, recons, pred_masks, gt_masks=None, attns=None, attns_origin=None, 
                    pos_pred=None, pos_pred_loc=None, pos_existence_mask=None, colored_box=True):
    '''
        `image`: [B, 3, H, W]
        `recon_combined`: [B, 3, H, W]
        `recons`: [B, K, H, W, C]
        `pred_masks`: [B, K, H, W, 1]
        `gt_masks`: [B, K, H, W, 1]
        `attns`: (B, T, N_heads, N_in, K)
        `attns_origin`: (B, T, N_heads, N_in, K)
        `pos_pred`: (B, L, K, 2)
        `pos_existence_mask`: (B, L, K, 1)

    '''

    image = image[:1]
    recon_combined = recon_combined[:1]
    recons = recons[:1]
    pred_masks = pred_masks[:1]
    gt_masks = gt_masks[:1]
    attns = attns[:1] 

    _, _, H, W = image.shape
    _, T, _, _, K = attns.shape

    T = 1

    # get binarized masks 
    pred_mask_max_idxs = torch.argmax(pred_masks.squeeze(-1), dim=1)
    # `mask_max_idxs`: (1, H, W)
    pred_seg_masks = torch.zeros_like(pred_masks.squeeze(-1))
    pred_seg_masks[torch.arange(1)[:, None, None], pred_mask_max_idxs, torch.arange(H)[None, :, None], torch.arange(W)[None, None, :]] = 1.0
    pred_seg_masks = pred_seg_masks.unsqueeze(-1)
    # `binary_masks`: (1, K, H, W, 1)

    pad = (0, 0, 2, 2, 2, 2)

    # set colors
    mark_colors = torch.tensor(np.array([[139,69,19], [255, 0, 0], [0, 255, 0], [0, 255, 255], [255, 0, 255], [0, 0, 255]]), dtype=torch.float32) / 255.0 
    
    slot_colors = torch.tensor(np.array([[255, 0, 0], [255, 127, 0], [255, 255, 0], [0, 255, 0], 
                                         [0, 0, 255], [75, 0, 130], [148, 0, 211], 
                                         [0, 255, 255], [153, 255, 153], [255, 153, 204], [102, 0, 51],
                                         [255, 255, 255], [0, 0, 0], [128, 128, 128]]), dtype=torch.float32) / 255.0 

    # handle the multi-head attention
    if T == 1:
        attns = torch.mean(attns[:, -1:], dim=2).permute(0, 1, 3, 2).view(1, 1, K, H, W).unsqueeze(-1)
    else: 
        attns = torch.mean(attns[:, :], dim=2).permute(0, 1, 3, 2).view(1, -1, K, H, W).unsqueeze(-1)
    # `attns`: (1, T, K, H, W, 1)
    if attns_origin is not None: 
        attns_origin = attns_origin[:1]
        if T == 1:
            attns_origin = torch.mean(attns_origin[:, -1:], dim=2).permute(0, 1, 3, 2).view(1, 1, K, H, W).unsqueeze(-1)
        else: 
            attns_origin = torch.mean(attns_origin[:, :], dim=2).permute(0, 1, 3, 2).view(1, -1, K, H, W).unsqueeze(-1)
        # `attns_origin`: (1, T, K, H, W, 1)

        attns_template = torch.zeros(1, T*2, K, H, W, 1, dtype=torch.float32, device=attns.device) 
        attns_template[:, torch.arange(0, T*2, 2)] = attns_origin
        attns_template[:, torch.arange(1, T*2, 2)] = attns
        attns = attns_template
        # `attns`: (1, T*2, K, H, W, 1)

    # reshape tensors
    attns = attns.reshape(-1, K, H, W, 1)
    # `attns`: (2, K, H, W, 1)
    
    attns = torch.cat([attns, pred_masks, pred_seg_masks], dim=0) 
    N_row = attns.shape[0]
    # `attns`: (N_row, K, H, W, 1)
    
    image = torch.einsum('nchw->nhwc', image)
    # image = torch.repeat_interleave(image, (T*2) if attns_origin is None else (T*2)*2, dim=0) # to vis. colored recon for attn_origin
    image_template = torch.ones((N_row, H, W, 3), dtype=image.dtype, device=image.device)
    image_template[-2] = 0 # to contain seg. mask by adding values
    if T == 1:
        image_template[-1] = image
    
    # draw boundary box for the original image
    image_template = F.pad(image_template, pad=pad, mode='constant', value=1.0)
    # if colored_box:
    #     image_template[:, :2, :] = image_template[:, -2:, :] = image_template[:, :, :2] = image_template[:, :, -2:] = slot_colors[-1]
    image_template = F.pad(image_template, pad=pad, mode='constant', value=1.0)
    # `image`: [(T*2)*2, H, W, C]


    recon_combined = torch.einsum('nchw->nhwc', recon_combined)
    recon_comb_template = torch.ones((N_row, H, W, 3), dtype=recon_combined.dtype, device=recon_combined.device)
    if T > 1:
        recon_comb_template[0] = image
    recon_comb_template[-2] = 0 # to contain seg. mask by adding values
    recon_comb_template[-1] = recon_combined

    # draw boundary box for the reconstructed image
    recon_comb_template = F.pad(recon_comb_template, pad=pad, mode='constant', value=1.0)
    # if colored_box:
    #     recon_combined[:, :2, :] = recon_combined[:, -2:, :] = recon_combined[:, :, :2] = recon_combined[:, :, -2:] = slot_colors[-2]
    recon_comb_template = F.pad(recon_comb_template, pad=pad, mode='constant', value=1.0)

    # get idxs to draw mark for `pos_pred`
    if pos_pred is not None:
        pos_pred = pos_pred[:1]
        _, L_pred, _, _ = pos_pred.shape

        # denormalize `pos_pred`
        denormed_pos_pred = torch.stack([W * (pos_pred[..., 0] + 0.5), H * (pos_pred[..., 1] + 0.5)], dim=-1)
        # `denormed_pos_pred`: (N_samples, L_pred, K, 2)

        # get the position to draw mark
        mark_start = denormed_pos_pred - 3
        mark_start = torch.where(mark_start > 0, mark_start, torch.zeros_like(mark_start, dtype=torch.float32, device=mark_start.device))
        mark_start = mark_start.long()
        # `mark_start`: (N_samples, L_pred, K, 2)

        mark_end = denormed_pos_pred + 3
        mark_end[..., 0] = torch.where(mark_end[..., 0] < W, mark_end[..., 0], torch.zeros_like(mark_end[..., 0], dtype=torch.float32, device=mark_end.device))
        mark_end[..., 1] = torch.where(mark_end[..., 1] < H, mark_end[..., 1], torch.zeros_like(mark_end[..., 1], dtype=torch.float32, device=mark_end.device))
        mark_end = mark_end.long()
        # `mark_end`: (N_samples, L_pred, K, 2)

    if pos_existence_mask is not None: 
        pos_existence_mask = pos_existence_mask[:1]
        # `pos_existence_mask`: (N_samples, K, 1)

    # log_img = torch.cat([image_template, recon_comb_template], dim=2)
    for k in range(K):

        # # get vis. of attention maps based on the original image
        picture = torch.ones_like(image) * attns[:, k, :, :, :]

        # overwrite vis. of alpha mask with the recon. image
        picture[-2] = pred_seg_masks[:, k, :, :, :]        
        picture[-1] = recons[:, k, :, :, :] * attns[-1, k, :, :, :] + (1 - attns[-1, k, :, :, :])

        recon_comb_template[-2, 4:-4, 4:-4, :] += pred_seg_masks[0, k, :, :, :] * slot_colors[k].to(pred_seg_masks.device)
        image_template[-2, 4:-4, 4:-4, :] += gt_masks[0, k, :, :, :] * slot_colors[k-1].to(pred_seg_masks.device) # `k-1` -> to give white color to background

        if pos_pred is not None: 
            for enum_idx, l in enumerate(pos_pred_loc):

                # draw mark for each iteration on the corresponding attn map 
                picture[2*l-1, mark_start[0, enum_idx, k, 1]:mark_end[0, enum_idx, k, 1], mark_start[0, enum_idx, k, 0]:mark_end[0, enum_idx, k, 0], :] = mark_colors[l]

                # draw mark for all iterations on the alpha mask 
                # picture[(T+2)+T, mark_start[0, enum_idx, k, 1]:mark_end[0, enum_idx, k, 1], mark_start[0, enum_idx, k, 0]:mark_end[0, enum_idx, k, 0], :] = mark_colors[l]

        # draw boundary box for slots
        picture = F.pad(picture, pad=pad, mode='constant', value=1.0)
        if colored_box:
            picture[:, :2, :] = picture[:, -2:, :] = picture[:, :, :2] = picture[:, :, -2:] = slot_colors[k]
        picture = F.pad(picture, pad=pad, mode='constant', value=1.0)

        if k == 0:
            log_img = torch.cat([picture], dim=2)
        else: 
            log_img = torch.cat([log_img, picture], dim=2)

    bg_mask = torch.where(torch.sum(image_template[-2, 4:-4, 4:-4, :], dim=-1, keepdim=True) == 0, torch.ones_like(image_template[-2, 4:-4, 4:-4, :]), torch.zeros_like(image_template[-2, 4:-4, 4:-4, :]))
    image_template[-2, 4:-4, 4:-4, :] += bg_mask * 0.5
    if T == 1:
        log_img = torch.cat([image_template, recon_comb_template, log_img], dim=2)
    else: 
        log_img = torch.cat([recon_comb_template, log_img], dim=2)

    log_img = log_img.permute(0, 3, 1, 2)
    return log_img
