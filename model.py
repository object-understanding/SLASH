import functools
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from scipy.optimize import linear_sum_assignment


# Kernel Normalized Kernel
class WNConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=False, norm_type="softmax", use_normed_logits=False):
        super(WNConv, self).__init__(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
        self.norm_type = norm_type

    def forward(self, input):
        o_c, i_c, k1, k2 = self.weight.shape
        weight = torch.reshape(self.weight.data, (o_c, i_c, k1 * k2))
        weight = weight / torch.linalg.norm(weight, dim=-1, keepdim=True) # norm logits to 1.

        if 'linear' in self.norm_type:
            weight = weight / torch.sum(weight, dim=-1, keepdim=True)
        elif 'softmax' in self.norm_type:
            weight = F.softmax(weight, dim=-1)

        self.weight = nn.Parameter(torch.reshape(weight, (o_c, i_c, k1, k2)))

        # we don't recommend ver lower than 1.7
        if '1.7' in torch.__version__:
            return self._conv_forward(input, self.weight)
        else:
            return self._conv_forward(input, self.weight, self.bias)
        


class SlotAttention(nn.Module):
    def __init__(self, cfg, eps=1e-8, device=None):
        super().__init__()
        """Builds the Slot Attention module
        Args:
        num_slots (K): Number of slots in Slot Attention.
        iterations: Number of iterations in Slot Attention.
        """
        self.device = device 
        self.num_slots = cfg.MODEL.SLOT.NUM
        self.iterations = cfg.MODEL.SLOT.ITERATIONS
        self.num_heads = cfg.MODEL.SLOT.ATTN_HEADS
        self.hid_dim = cfg.MODEL.HID_DIM
        self.slot_dim = cfg.MODEL.SLOT.DIM
        self.mlp_hid_dim = cfg.MODEL.MLP_HID_DIM
        self.scale = (cfg.MODEL.SLOT.DIM // cfg.MODEL.SLOT.ATTN_HEADS) ** -0.5
        self.weak_sup = cfg.WEAK_SUP.TYPE
        self.use_no_obj = cfg.WEAK_SUP.USE_NO_OBJ
        self.rand_pos_type = cfg.WEAK_SUP.RAND_POS_TYPE
        self.init_using_sup = cfg.WEAK_SUP.INIT_USING_SUP
        self.weak_sup_split_ratio = cfg.WEAK_SUP.SPLIT.RATIO
        self.temperature = cfg.MODEL.SLOT.TEMPERATURE
        self.slot_self_attn = cfg.MODEL.SLOT.SELF_ATTN
        self.slot_attn_smooth = cfg.MODEL.SLOT.ATTN_SMOOTH

        self.use_pos_pred = cfg.POS_PRED.USE_POS_PRED
        self.pos_pred_use_gt = cfg.POS_PRED.USE_GT # TODO: name of variable?
        self.pos_pred_use_no_obj = cfg.POS_PRED.USE_NO_OBJ
        self.pos_pred_location_list = cfg.POS_PRED.LOCATIONS

        self.use_batch_fusion = cfg.WEAK_SUP.SPLIT.TRAIN.MODE == 'batch_fusion'
        self.batch_fusion_ratio = cfg.WEAK_SUP.SPLIT.TRAIN.BATCH_FUSION_RATIO
        self.batch_fusion_ws_num_samples = int(cfg.TRAIN.BATCH_SIZE * self.batch_fusion_ratio) 
        if not cfg.TRAIN.IS_DISTRIBUTED:
            self.batch_fusion_ws_num_samples //= torch.cuda.device_count()

        self.feature_pyramid_length = len(cfg.MODEL.FEAT_PYRAMID.RES_RATE_LIST)
        self.pyramid_aggregation_mode = cfg.MODEL.FEAT_PYRAMID.AGGREGATION

        self.eps = eps

        assert (self.init_using_sup and self.weak_sup != '') or not self.init_using_sup
        if self.use_pos_pred:
            assert cfg.WEAK_SUP.TYPE != ""
            assert self.pos_pred_location_list != []

        if self.init_using_sup:
            # MLP to initialize slot
            # hidden layer follows the SAVi model (256 when slot_dim == 128)
            pos_num = 4 if self.weak_sup == 'bbox' else 2
            self.slot_initializer = nn.Sequential(
                nn.Linear(pos_num, self.slot_dim * 2), nn.ReLU(), nn.Linear(self.slot_dim * 2, self.slot_dim)
            )
            if self.rand_pos_type == 'standard_gaussian' and not self.use_no_obj:
                self.pos_mu = torch.zeros(1, 1, pos_num)
                self.pos_sigma = torch.ones(1, 1, pos_num)
            elif self.rand_pos_type == 'learnable_gaussian'and not self.use_no_obj:
                self.pos_mu = nn.Parameter(torch.randn(1, 1, pos_num))
                self.pos_sigma = nn.Parameter(torch.abs(torch.randn(1, 1, pos_num)))
        else:
            # original slot initialize
            self.slots_mu = nn.Parameter(torch.randn(1, 1, self.slot_dim))
            self.slots_sigma = nn.Parameter(torch.abs(torch.randn(1, 1, self.slot_dim)))

        # self.norm_input = nn.LayerNorm(self.hid_dim)
        self.norm_input = nn.ModuleList()
        for _ in range(self.feature_pyramid_length):
            self.norm_input.append(nn.LayerNorm(self.hid_dim))
        self.norm_slots = nn.LayerNorm(self.slot_dim)
        self.norm_mlp = nn.LayerNorm(self.slot_dim)

        self.to_q = nn.Linear(self.slot_dim, self.slot_dim)
        # self.to_k = nn.Linear(self.hid_dim, self.slot_dim)
        # self.to_v = nn.Linear(self.hid_dim, self.slot_dim)

        self.to_k = nn.ModuleList()
        self.to_v = nn.ModuleList()
        for _ in range(self.feature_pyramid_length):
            self.to_k.append(nn.Linear(self.hid_dim, self.slot_dim))
            self.to_v.append(nn.Linear(self.hid_dim, self.slot_dim))

        if self.slot_self_attn:
            self.multihead_attn = nn.MultiheadAttention(self.slot_dim, self.num_heads, batch_first=True)
            self.self_attn_to_q = nn.Linear(self.slot_dim, self.slot_dim)
            self.self_attn_to_k = nn.Linear(self.slot_dim, self.slot_dim)
            self.self_attn_to_v = nn.Linear(self.slot_dim, self.slot_dim)


        if self.slot_attn_smooth != '':
            # TODO: (N to 1) or (N to N) conv may be possible (now 1 to 1)
            # TODO: change name knconv to ask
            kernel_size = cfg.MODEL.SLOT.ATTN_SMOOTH_SIZE
            if self.slot_attn_smooth == 'gaussian':
                sigma_min = cfg.MODEL.SLOT.ATTN_SMOOTH_GAU_MIN
                sigma_max = cfg.MODEL.SLOT.ATTN_SMOOTH_GAU_MAX
                self.knconv = torchvision.transforms.GaussianBlur(kernel_size=kernel_size, 
                                                               sigma=(sigma_min, sigma_max))
            elif self.slot_attn_smooth == 'conv':
                self.knconv = nn.Conv2d(in_channels=1,
                                     out_channels=1,
                                     kernel_size=kernel_size,
                                     padding=kernel_size // 2,
                                     bias=False)
            elif self.slot_attn_smooth == 'wnconv':
                self.knconv = WNConv(in_channels=1,
                                    out_channels=1,
                                    kernel_size=kernel_size,
                                    padding=kernel_size // 2,
                                    bias=False)

        self.gru = nn.GRUCell(self.slot_dim, self.slot_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.slot_dim, self.mlp_hid_dim), nn.ReLU(), nn.Linear(self.mlp_hid_dim, self.slot_dim)
        )

        if self.use_pos_pred:
            self.pos_predictor = PositionPredictor(cfg)
            self.pos_encoder = PositionEncoder(cfg)

    def predict_and_encode_position(self, slots, pos, pos_gt_aranged, pos_emb, train=False):
        assert slots is not None 

        pos_pred = None
        if not train or (train and self.weak_sup_split_ratio >= 1 - 1e-4):
            if self.pos_pred_use_gt and pos != None:
                pos_pred = self.pos_predictor(slots.detach())
                # pos_pred = self.pos_predictor(slots.clone()) # [B, K (or K-1), 2]

                if pos_gt_aranged is None:
                    # matching gt to pred
                    # cal cost map to match gt to pred
                    cost_map = torch.cdist(pos_pred, pos).cpu().detach().numpy() # [B, K, K]
                    # match gt and pred using linear sum assignment
                    # sanity check done by toy examples
                    match_indexes = np.array([linear_sum_assignment(cost_map[i])[1] for i in range(len(pos))]).reshape(-1) # [B*K,]
                    batch_index = [i // pos.shape[1] for i in range(pos.shape[0] * pos.shape[1])]
                    pos_gt_aranged = pos[range(len(pos))][batch_index, match_indexes].reshape(pos.shape)
                
                if not self.pos_pred_use_no_obj:
                    # fill pos_pred value for invalid matching
                    pos_pred_with_gt = torch.where(pos_gt_aranged > -1, pos_gt_aranged.type(pos_pred.dtype), pos_pred)
                
                pos_encoder_input = pos_pred_with_gt.clone() # [B, K (or K-1), 2]

                pos_encoder_input = pos_encoder_input.detach()
                pos_encoded_slots = self.pos_encoder(pos_encoder_input).to(self.device)
                
                slots = slots + pos_encoded_slots

            else:
                pos_pred = self.pos_predictor(slots.detach())
                # pos_pred = self.pos_predictor(slots.clone())

                pos_encoder_input = pos_pred.detach().clone()
                pos_encoded_slots = self.pos_encoder(pos_encoder_input).to(self.device)
                
                slots = slots + pos_encoded_slots

        elif self.use_batch_fusion and train:
            pos_pred = self.pos_predictor(slots.detach())

            # extract pos_pred_for_samples_wo_sup and pos_for_samples_wo_sup 
            # so that they don't participate in gt matching
            pos_pred_for_samples_wo_sup = pos_pred[self.batch_fusion_ws_num_samples:]
            pos_pred = pos_pred[:self.batch_fusion_ws_num_samples]
            pos_for_samples_wo_sup = pos[self.batch_fusion_ws_num_samples:]
            pos = pos[:self.batch_fusion_ws_num_samples]

            if pos_gt_aranged is None:
                # matching gt to pred
                cost_map = torch.cdist(pos_pred, pos).cpu().detach().numpy() # [B, num_w_sup, num_w_sup]
                match_indexes = np.array([linear_sum_assignment(cost_map[i])[1] for i in range(len(pos))]).reshape(-1) # [B*K,]
                batch_index = [i // pos.shape[1] for i in range(pos.shape[0] * pos.shape[1])]
                pos_gt_aranged = pos[range(len(pos))][batch_index, match_indexes].reshape(pos.shape)

            if not self.pos_pred_use_no_obj:
                # fill pos_pred value for invalid matching
                pos_pred_with_gt = torch.where(pos_gt_aranged > -1, pos_gt_aranged.type(pos_pred.dtype), pos_pred)

            # glue pos_pred_for_samples_wo_sup and pos_for_samples_wo_sup 
            # to make original size pos_pred and pos
            pos_pred = torch.cat([pos_pred, pos_pred_for_samples_wo_sup], dim=0)
            pos = torch.cat([pos, pos_for_samples_wo_sup], dim=0)

            if self.pos_pred_use_gt:
                pos_encoder_input = pos_pred.clone()
                pos_encoder_input[:self.batch_fusion_ws_num_samples] = pos_pred_with_gt.clone()
            else:
                pos_encoder_input = pos_pred.clone()
            
            pos_encoder_input = pos_encoder_input.detach()
            pos_encoded_slots = self.pos_encoder(pos_encoder_input).to(self.device)
            
            slots = slots + pos_encoded_slots

        outputs = dict()
        outputs["pos_pred"] = pos_pred
        outputs["pos_gt_aranged"] = pos_gt_aranged
        outputs["slots"] = slots
        return outputs
             

    def forward(self, inputs_list, num_slots=None, pos=None, pos_emb=None, train=False):
        outputs = dict()
        outputs["pos_pred"] = []

        # B, N_in, D_in = inputs.shape
        B = inputs_list[0].shape[0]
        N_in_list = []
        D_in_list = []
        upscale_factor_list = []
        for inputs in inputs_list:
            N_in_list.append(inputs.shape[1])
            D_in_list.append(inputs.shape[2])
            upscale_factor_list.append(int(np.sqrt(N_in_list[0] // N_in_list[-1])))
        
        K = num_slots if num_slots is not None else self.num_slots
        D_slot = self.slot_dim
        N_heads = self.num_heads
    
        if self.init_using_sup:
            if pos is None:
                pos = -1 * torch.ones(B, K, 2).to(self.device)
            if not self.use_no_obj:
                if self.rand_pos_type == 'uniform':
                    rand_pos = torch.rand(pos.shape) - 0.5
                elif self.rand_pos_type in ['standard_gaussian', 'learnable_gaussian']:
                    pos_mu = self.pos_mu.expand(B, K, -1)
                    pos_sigma = torch.abs(self.pos_sigma.expand(B, K, -1))
                    rand_pos = torch.normal(pos_mu, pos_sigma)
                if self.use_batch_fusion:
                    pos[self.batch_fusion_ws_num_samples:] = -1.
                pos = torch.where(pos > -1, pos, rand_pos.to(self.device))
            slots = self.slot_initializer(pos)
        else:
            # original initialize of slots
            mu = self.slots_mu.expand(B, K, -1)
            sigma = torch.abs(self.slots_sigma.expand(B, K, -1))
            slots = torch.normal(mu, sigma + self.eps)

        # inputs = self.norm_input(inputs)
        # # k, v: (B, N_heads, num_inputs, slot_dim // N_heads).
        # k = self.to_k(inputs).reshape(B, N_in, N_heads, -1).transpose(1, 2)
        # v = self.to_v(inputs).reshape(B, N_in, N_heads, -1).transpose(1, 2)
        k_list = []
        v_list = []
        for input_i in range(len(inputs_list)):
            inputs_list[input_i] = self.norm_input[input_i](inputs_list[input_i])

            k_list.append(self.to_k[input_i](inputs_list[input_i]).reshape(B, N_in_list[input_i], N_heads, -1).transpose(1, 2))
            v_list.append(self.to_v[input_i](inputs_list[input_i]).reshape(B, N_in_list[input_i], N_heads, -1).transpose(1, 2))
    
        pos_gt_aranged = None

        if not train:
            attns = list()
            attns_origin = list()

        if 0 in self.pos_pred_location_list:
            pp_outputs = self.predict_and_encode_position(slots, pos, pos_gt_aranged, pos_emb, train)
            outputs["pos_pred"].append(pp_outputs["pos_pred"]) 
            pos_gt_aranged = pp_outputs["pos_gt_aranged"]
            slots = pp_outputs["slots"]

        for iter_i in range(self.iterations):
            slots_prev = slots

            if self.slot_self_attn: 
                self_attn_q = (self.self_attn_to_q(slots))  # (B, K, slot_D)
                self_attn_k = (self.self_attn_to_k(slots))  # (B, K, slot_D)
                self_attn_v = (self.self_attn_to_v(slots))  # (B, K, slot_D)
                slots, _ = self.multihead_attn(self_attn_q, self_attn_k, self_attn_v)

            slots = self.norm_slots(slots)

            q = (self.to_q(slots).reshape(B, K, N_heads, -1).transpose(1, 2))  
            # `q`: (B, N_heads, K, slot_D // N_heads)

            attn_list = [] 
            attn_origin_list = [] 
            for input_i in range(len(inputs_list)):
                k = k_list[input_i]

                attn_logits = (torch.einsum('bhid,bhjd->bhij', k, q) * self.scale)  
                attn_origin = (attn_logits / self.temperature).softmax(dim=-1) + self.eps  # Normalization over slots

                if self.slot_attn_smooth != '':
                    # TODO: this part can go to 
                    # after softmax of attn_logits
                    attn_logits = attn_logits.permute(0, 3, 1, 2) # [B, K, N_heads, N_in]
                    attn_logits = attn_logits.reshape(-1, N_in_list[input_i])[:, None, :] # [B*K*N_head, 1, N_in]
                    img_size = int(N_in_list[input_i] ** 0.5)
                    attn_logits = attn_logits.reshape(-1, 1, img_size, img_size) # [B*K*N_heads, 1, img_size, img_size]
                    attn_logits = self.knconv(attn_logits) # [B*K*N_heads, 1, img_size, img_size]
                    attn_logits = attn_logits.reshape(B, K, N_heads, N_in_list[input_i]) # [B, K, N_heads, N_in]
                    attn_logits = attn_logits.permute(0, 2, 3, 1) # (B, N_heads, N_in, K)

                attn = (attn_logits / self.temperature).softmax(dim=-1) + self.eps  # Normalization over slots
                # `attn`: (B, N_heads, N_in, K)

                attn_list.append(attn) 
                attn_origin_list.append(attn_origin)

            if self.pyramid_aggregation_mode == 'attn': 
                # upsample attn to the original input size 
                upscaled_attn_list = [] 
                upscaled_attn_origin_list = []
                for attn_i, attn in enumerate(attn_list):
                    upscale_factor = upscale_factor_list[attn_i]
                    hw = N_in_list[attn_i]
                    h = w = int(np.sqrt(hw))

                    attn = attn.reshape(B, N_heads, h, w, K)
                    attn = torch.repeat_interleave(attn, upscale_factor, dim=2)
                    attn = torch.repeat_interleave(attn, upscale_factor, dim=3)
                    attn = attn.reshape(B, N_heads, N_in_list[0], K)
                    # `attn`: (B, N_heads, N_in, K)

                    attn_origin = attn_origin_list[attn_i]
                    attn_origin = attn_origin.reshape(B, N_heads, h, w, K)
                    attn_origin = torch.repeat_interleave(attn_origin, upscale_factor, dim=2)
                    attn_origin = torch.repeat_interleave(attn_origin, upscale_factor, dim=3)
                    attn_origin = attn_origin.reshape(B, N_heads, N_in_list[0], K)
                    # `attn_origin`: (B, N_heads, N_in, K)

                    upscaled_attn_list.append(attn)
                    upscaled_attn_origin_list.append(attn_origin)
                
                # aggregation `upscaled_attn_list`
                attn = torch.mean(torch.stack(upscaled_attn_list, dim=0), dim=0)
                attn_origin = torch.mean(torch.stack(upscaled_attn_origin_list, dim=0), dim=0)
                if not train:
                    attns.append(attn)
                    attns_origin.append(attn_origin)
                attn = attn / torch.sum(attn, dim=-2, keepdim=True)  # Weigthed mean
                attns_origin = attns_origin / torch.sum(attn, dim=-2, keepdim=True)  # Weigthed mean
                # `attn`: (B, N_heads, N_in, K)
                # `attns_origin`: (B, N_heads, N_in, K)

                v = v_list[0] 
                # `v`: (B, N_heads, N_in, slot_D // N_heads)

                updates = torch.einsum('bhij,bhid->bhjd', attn, v)  
                # `updates`: (B, N_heads, K, slot_D // N_heads)
                updates = updates.transpose(1, 2).reshape(B, K, -1)  
                # `updates`: (B, K, slot_D)
                
            elif self.pyramid_aggregation_mode == 'slot':  
                if not train:
                    attns.append(attn_list[0])
                    attns_origin.append(attn_origin_list[0])

                updates_list = []
                for attn_i, attn in enumerate(attn_list):
                    v = v_list[attn_i]
                    # `v`: (B, N_heads, N_in, slot_D // N_heads)

                    attn = attn / torch.sum(attn, dim=-2, keepdim=True)  # Weigthed mean
                    # `attn`: (B, N_heads, N_in, K)
                    
                    updates = torch.einsum('bhij,bhid->bhjd', attn, v)  
                    # `updates`: (B, N_heads, K, slot_D // N_heads)
                    updates = updates.transpose(1, 2).reshape(B, K, -1)  
                    # `updates`: (B, K, slot_D)

                    updates_list.append(updates)
                updates = torch.mean(torch.stack(updates_list, dim=0), dim=0)
                # `updates`: (B, K, slot_D)

            slots = self.gru(
                updates.reshape(-1, self.slot_dim), slots_prev.reshape(-1, self.slot_dim)
            )

            slots = slots.reshape(B, -1, self.slot_dim)

            if iter_i + 1 in self.pos_pred_location_list:
                pp_outputs = self.predict_and_encode_position(slots, pos, pos_gt_aranged, pos_emb, train)
                outputs["pos_pred"].append(pp_outputs["pos_pred"]) 
                pos_gt_aranged = pp_outputs["pos_gt_aranged"]
                slots = pp_outputs["slots"]

            slots = slots + self.mlp(self.norm_mlp(slots))
    
        # predict position
        if -1 in self.pos_pred_location_list:
            pp_outputs = self.predict_and_encode_position(slots, pos, pos_gt_aranged, pos_emb, train)
            outputs["pos_pred"].append(pp_outputs["pos_pred"]) 
            pos_gt_aranged = pp_outputs["pos_gt_aranged"]
            slots = pp_outputs["slots"]

        outputs["slots"] = slots
        outputs["attn"] = attn
        outputs["pos_gt_aranged"] = pos_gt_aranged
        if not train: 
            outputs["attns"] = attns
            outputs["attns_origin"] = attns_origin
            # `attns`: list of (B, N_heads, N_in, K) x T

        return outputs

def build_grid(resolution):
    ranges = [np.linspace(0.0, 1.0, num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1))


"""Adds soft positional embedding with learnable projection."""
class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution, device):
        """Builds the soft position embedding layer.
        Args:
        hidden_size: Size of input feature dimension.
        resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.device = device
        self.embedding = nn.Linear(4, hidden_size, bias=True)
        self.grid = build_grid(resolution)

    def forward(self, inputs):
        # self.grid = self.grid.to(inputs.device)
        self.grid = self.grid.to(self.device)
        grid = self.embedding(self.grid)
        return inputs + grid

    def get_pos_emb(self, pos):
        """ 
        pos (*, 2)
        """
        return self.embedding(torch.cat([pos, 1.0 - pos], dim=-1))


class Encoder(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        assert cfg.MODEL.ENC_DEPTH > 2, "Depth must be larger than 2."

        resolution = (cfg.DATA.IMG_SIZE, cfg.DATA.IMG_SIZE)
        hid_dim = cfg.MODEL.HID_DIM
        enc_depth = cfg.MODEL.ENC_DEPTH 
        feature_pyramid_res_rate_list = cfg.MODEL.FEAT_PYRAMID.RES_RATE_LIST
        self.feature_pyramid_res_rate_list = feature_pyramid_res_rate_list # include original res

        # convs = [nn.Conv2d(3, hid_dim, 5, padding='same'), nn.ReLU()]
        # for _ in range(enc_depth - 2):
        #     convs.extend([nn.Conv2d(hid_dim, hid_dim, 5, padding='same'), nn.ReLU()])
        # convs.append(nn.Conv2d(hid_dim, hid_dim, 5, padding='same'))

        root_conv = [nn.Conv2d(3, hid_dim, 5, padding=2), nn.ReLU()]
        for _ in range(enc_depth - 2):
            root_conv.extend([nn.Conv2d(hid_dim, hid_dim, 5, padding=2), nn.ReLU()])
        self.root_conv = nn.Sequential(*root_conv)

        self.convs_list = nn.ModuleList() 
        for res_rate in self.feature_pyramid_res_rate_list:
            conv = []
            for i in range(int(np.log2(1 / res_rate))):
                conv.append(nn.Conv2d(hid_dim, hid_dim, 5, stride=2, padding=2))
            conv.append(nn.Conv2d(hid_dim, hid_dim, 5, padding=2))
            conv = nn.Sequential(*conv)
            self.convs_list.append(conv)

        self.device = device
        
        self.encoder_pos_list = nn.ModuleList()
        self.layer_norm_list = nn.ModuleList()
        self.mlp_list = nn.ModuleList()
        for res_rate in self.feature_pyramid_res_rate_list:
            res = (int(resolution[0] * res_rate), int(resolution[1] * res_rate))
            self.encoder_pos_list.append(SoftPositionEmbed(hid_dim, res, device))
            self.layer_norm_list.append(nn.LayerNorm([res[0] * res[1], hid_dim]))
            self.mlp_list.append(nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(), nn.Linear(hid_dim, hid_dim)))

    def forward(self, x):
        x = self.root_conv(x) # [B, D, H, W]
        x_list = []
        for i in range(len(self.feature_pyramid_res_rate_list)):
            x_i = self.convs_list[i](x) # [B, D, H, W]
            x_i = x_i.permute(0, 2, 3, 1) # [B, H, W ,D]
            x_i = self.encoder_pos_list[i](x_i)
            x_i = torch.flatten(x_i, 1, 2)
            x_i = self.layer_norm_list[i](x_i)
            x_i = self.mlp_list[i](x_i)

            x_list.append(x_i)

        # x = self.convs(x) # [B, D, H, W]
        # x = x.permute(0, 2, 3, 1) # [B, H, W ,D]

        # x = self.encoder_pos(x)
        # x = torch.flatten(x, 1, 2)
        # x = self.layer_norm(x)
        # x = self.mlp(x)

        # return x
        return x_list


class Decoder(nn.Module):

    def __init__(self, cfg, device):
        super().__init__()

        D_slot = cfg.MODEL.SLOT.DIM
        D_hid = cfg.MODEL.DEC_HID_DIM if cfg.MODEL.DEC_HID_DIM > 0 else cfg.MODEL.HID_DIM
        
        target_size = cfg.DATA.IMG_SIZE
        self.resolution = (target_size, target_size)
        self.dec_init_size = cfg.MODEL.DEC_INIT_SIZE
        dec_init_resolution = (self.dec_init_size, self.dec_init_size)
        self.decoder_pos = SoftPositionEmbed(cfg.MODEL.SLOT.DIM, dec_init_resolution, device)
        upsample_step = int(np.log2(target_size // self.dec_init_size))
        
        count_layer = 0 

        deconvs = nn.ModuleList()
        for _ in range(upsample_step):
            if count_layer == 0: 
                deconvs.extend([
                    nn.ConvTranspose2d(D_slot, D_hid, 5, stride=(2, 2), padding=2, output_padding=1),
                    nn.ReLU(),
                ])
            else: 
                deconvs.extend([
                        nn.ConvTranspose2d(D_hid, D_hid, 5, stride=(2, 2), padding=2, output_padding=1),
                        nn.ReLU(),
                ])
            
            count_layer += 1

        for _ in range(cfg.MODEL.DEC_DEPTH - upsample_step - 1):
            
            if count_layer == 0: 
                deconvs.extend([nn.ConvTranspose2d(D_slot, D_hid, 5, stride=(1, 1), padding=2), nn.ReLU()])
            else: 
                deconvs.extend([nn.ConvTranspose2d(D_hid, D_hid, 5, stride=(1, 1), padding=2), nn.ReLU()])

            count_layer += 1

        deconvs.append(nn.ConvTranspose2d(D_hid, 4, 3, stride=(1, 1), padding=1))
        count_layer += 1

        assert cfg.MODEL.DEC_DEPTH == count_layer, "The number of layers of decoder differs from the configuration"

        self.deconvs = nn.Sequential(*deconvs)

    def forward(self, x):
        # """Broadcast slot features to a 2D grid and collapse slot dimension.""".
        x = x.reshape(-1, x.shape[-1]).unsqueeze(1).unsqueeze(2)
        x = x.repeat((1, self.dec_init_size, self.dec_init_size, 1))

        x = self.decoder_pos(x)
        x = x.permute(0, 3, 1, 2)
        x = self.deconvs(x)
        x = x[:, :, : self.resolution[0], : self.resolution[1]]
        x = x.permute(0, 2, 3, 1)
        return x


class PositionPredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.WEAK_SUP.TYPE != ""

        slot_dim = cfg.MODEL.SLOT.DIM
        pos_num = 4 if cfg.WEAK_SUP.TYPE == 'bbox' else 2
        self.layer_norm = nn.LayerNorm(slot_dim)
        if cfg.POS_PRED.PP_SIZE == 'base':
            self.mlp = nn.Sequential(
                nn.Linear(slot_dim, slot_dim // 2), 
                nn.ReLU(), 
                nn.Linear(slot_dim // 2, slot_dim // 4), 
                nn.ReLU(), 
                nn.Linear(slot_dim // 4, pos_num)
            )
        elif cfg.POS_PRED.PP_SIZE == 'small':
            self.mlp = nn.Sequential(
                nn.Linear(slot_dim, slot_dim // 4), 
                nn.ReLU(), 
                nn.Linear(slot_dim // 4, pos_num)
            )
        elif cfg.POS_PRED.PP_SIZE == 'big':
            self.mlp = nn.Sequential(
                nn.Linear(slot_dim, slot_dim), 
                nn.ReLU(), 
                nn.Linear(slot_dim, slot_dim // 4), 
                nn.ReLU(), 
                nn.Linear(slot_dim // 4, slot_dim // 8), 
                nn.ReLU(), 
                nn.Linear(slot_dim // 8, pos_num)
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.mlp(self.layer_norm(x))) - 0.5


class PositionEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        slot_dim = cfg.MODEL.SLOT.DIM
        pos_num = 4 if cfg.WEAK_SUP.TYPE == 'bbox' else 2
        self.encoder = nn.Sequential(
            nn.Linear(pos_num, slot_dim // 4), 
            nn.ReLU(), 
            nn.Linear(slot_dim // 4, slot_dim // 2), 
            nn.ReLU(), 
            nn.Linear(slot_dim // 2, slot_dim)
        )
            
    def forward(self, x):
        return self.encoder(x)
        

"""Slot Attention-based auto-encoder for object discovery."""
class SlotAttentionAutoEncoder(nn.Module):
    def __init__(self, cfg, device):
        # TODO: update below docstring
        """Builds the Slot Attention-based auto-encoder.
        Args:
        num_slots (K): Number of slots in Slot Attention.
        """
        super().__init__()
        self.device = device
        self.slot_dim = cfg.MODEL.SLOT.DIM

        self.encoder_cnn = Encoder(cfg, device=device)
        self.decoder_cnn = Decoder(cfg, device=device)

        self.slot_attention = SlotAttention(cfg=cfg, eps=1e-8, device=device)

    def forward(self, image, pos=None, train=True):
        # `image` has shape: [batch_size, num_channels, height, width].
        
        # Convolutional encoder with position embedding.
        # x = self.encoder_cnn(image)  # CNN Backbone.
        x_list = self.encoder_cnn(image)  # CNN Backbone.

        # `x` has shape: [B, height*width, hid_dim].

        # Slot Attention module.
        # sa_outputs = self.slot_attention(x, pos=pos, train=train)
        sa_outputs = self.slot_attention(x_list, pos=pos, pos_emb=self.encoder_cnn.encoder_pos_list[-1], train=train)
        slots = sa_outputs["slots"]
        # `slots` has shape: [N, K, slot_dim].

        x = self.decoder_cnn(slots)

        # `x` has shape: [B*K, height, width, num_channels+1].

        # Undo combination of slot and batch dimension; split alpha masks.
        recons, masks = x.reshape(image.shape[0], -1, x.shape[1], x.shape[2], x.shape[3]
                        ).split([3, 1], dim=-1)
        # `recons` has shape: [B, K, height, width, num_channels].
        # `masks` has shape: [B, K, height, width, 1].

        # Normalize alpha masks over slots.
        masks = nn.Softmax(dim=1)(masks)

        recon_combined = torch.sum(recons * masks, dim=1)  # Recombine image.
        recon_combined = recon_combined.permute(0, 3, 1, 2)
        # `recon_combined` has shape: [batch_size, num_channels, height, width].

        outputs = dict()
        outputs['recon_combined'] = recon_combined
        outputs['recons'] = recons
        outputs['masks'] = masks
        outputs['slots'] = slots
        outputs['attn'] = sa_outputs["attn"]
        outputs['pos_pred'] = sa_outputs["pos_pred"]
        outputs['pos_gt_aranged'] = sa_outputs['pos_gt_aranged']
        if not train: 
            outputs['attns'] = sa_outputs['attns']
            outputs['attns_origin'] = sa_outputs['attns_origin']
            # `attns`: list of (B, N_heads, N_in, K) x T

        return outputs

