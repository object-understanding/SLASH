import torch 
import torch.nn.functional as F 
import numpy as np 

def visualize(image, recon_combined, recons, masks, attns=None, pos_pred=None, num_vis=4, colored_box=True):
    '''
        `image`: [B, 3, H, W]
        `recon_combined`: [B, 3, H, W]
        `recons`: [B, K, H, W, C]
        `masks`: [B, K, H, W, 1]
        `attns`: (B, T, N_heads, N_in, K)
        `pos_pred`: (B, L, K, 2)

    '''

    N_samples = num_vis

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
                for iter_idx in range(T):

                    # draw mark for each iteration on the corresponding attn map 
                    picture[n*(T+2)+iter_idx, mark_start[n, iter_idx, k, 1]:mark_end[n, iter_idx, k, 1], mark_start[n, iter_idx, k, 0]:mark_end[n, iter_idx, k, 0], :] = mark_colors[iter_idx+1]

                    # draw mark for all iterations on the alpha mask 
                    picture[n*(T+2)+T, mark_start[n, iter_idx, k, 1]:mark_end[n, iter_idx, k, 1], mark_start[n, iter_idx, k, 0]:mark_end[n, iter_idx, k, 0], :] = mark_colors[iter_idx+1]

        # draw boundary box for slots
        picture = F.pad(picture, pad=pad, mode='constant', value=1.0)
        if colored_box:
            picture[:, :2, :] = picture[:, -2:, :] = picture[:, :, :2] = picture[:, :, -2:] = slot_colors[k]
        picture = F.pad(picture, pad=pad, mode='constant', value=1.0)

        log_img = torch.cat([log_img, picture], dim=2)

    log_img = log_img.permute(0, 3, 1, 2)
    return log_img
