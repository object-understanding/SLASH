import torch 
import numpy as np 
from scipy.special import comb
from scipy.optimize import linear_sum_assignment

class ARIEvaluator():
    def __init__(self):
        self.aris = []
        
    def evaluate(self, pred, label, device):
        """
        :param data: (image, mask)
            image: (B, 3, H, W)
            pred : (B, N0, H, W)
            label: (B, N1, H, W)
        :return: average ari
        """
        from torch import arange as ar

        pred = pred[:, :, :, :, 0] # pred: (B, K, H, W, 1) -> (B, K, H, W)
        
        B, K, H, W = pred.size()
        
        # reduced to (B, K, H, W), with 1-0 values
        
        # max_index (B, H, W)
        max_index = torch.argmax(pred, dim=1)
        # get binarized masks (B, K, H, W)
        pred = torch.zeros_like(pred)
        pred[ar(B)[:, None, None], max_index, ar(H)[None, :, None], ar(W)[None, None, :]] = 1.0

        for b in range(B):
            this_ari = self.compute_mask_ari(label[b].to(device), pred[b].to(device))
            self.aris.append(this_ari)
        
    
    def reset(self):
        self.aris = []
    
    def get_results(self):
        return np.mean(self.aris) if self.aris != [] else 0
    
    def compute_ari(self, table):
        """
        Compute ari, given the index table
        :param table: (r, s)
        :return:
        """
        
        # # (r,)
        # a = table.sum(axis=1)
        # # (s,)
        # b = table.sum(axis=0)
        # n = a.sum()
        # (r,)
        a = table.sum(dim=1)
        # (s,)
        b = table.sum(dim=0)
        n = a.sum()
        
        comb_a = comb(a.detach().cpu().numpy(), 2).sum()
        comb_b = comb(b.detach().cpu().numpy(), 2).sum()
        comb_n = comb(n.detach().cpu().numpy(), 2)
        comb_table = comb(table.detach().cpu().numpy(), 2).sum()
        
        if (comb_b == comb_a == comb_n == comb_table):
            # the perfect case
            ari = 1.0
        else:
            ari = (
                (comb_table - comb_a * comb_b / comb_n) /
                (0.5 * (comb_a + comb_b) - (comb_a * comb_b) / comb_n)
            )
        
        return ari
        
    def compute_mask_ari(self, mask0, mask1):
        """
        Given two sets of masks, compute ari
        :param mask0: ground truth mask, (N0, H, W)
        :param mask1: predicted mask, (N1, H, W)
        :return:
        """
        
        # will first need to compute a table of shape (N0, N1)
        # (N0, 1, H, W)
        mask0 = mask0[:, None].byte()
        # (1, N1, H, W)
        mask1 = mask1[None, :].byte()
        # (N0, N1, H, W)
        agree = mask0 & mask1
        # (N0, N1)
        table = agree.sum(dim=-1).sum(dim=-1)
        
        return self.compute_ari(table)
        

class mBOEvaluator():
    def __init__(self):
        self.mbos = []

    def evaluate(self, pred, label, device):
        """
        :param data: (image, mask)
            image: (B, 3, H, W)
            pred : (B, N0, H, W)
            label: (B, N1, H, W)
        :return: average mbo
        """
        from torch import arange as ar

        pred = pred[:, :, :, :, 0] # pred: (B, K, H, W, 1) -> (B, K, H, W)
        
        B, K, H, W = pred.size()
        
        # reduced to (B, K, H, W), with 1-0 values
        
        # max_index (B, H, W)
        max_index = torch.argmax(pred, dim=1)
        # get binarized masks (B, K, H, W)
        pred = torch.zeros_like(pred)
        pred[ar(B)[:, None, None], max_index, ar(H)[None, :, None], ar(W)[None, None, :]] = 1.0

        for b in range(B):
            this_mbo = self.compute_mbo(label[b].to(device), pred[b].to(device))
            self.mbos.append(this_mbo)

    def reset(self):
        self.mbos = []
    
    def get_results(self):
        return np.mean(self.mbos) if self.mbos != [] else 0

    def compute_mbo(self, mask0, mask1):
        """
        mask0: [N0, H, W]
        mask1: [N1, H, W]
        """

        mask0 = mask0.reshape(mask0.shape[0], -1)[:, None, :] # [N0, 1, H*W]
        mask1 = mask1.reshape(mask1.shape[0], -1)[None, :, :] # [1, N1, H*W]

        union = torch.sum(torch.clip(mask0 + mask1, 0, 1), dim=-1) # [1, N0, N1]
        intersection = torch.sum(mask0 * mask1, dim=-1) # [1, N0, N1]
        iou = (intersection / (union + 1e-8)) # [N0, N1]

        row_ind, col_ind = linear_sum_assignment(iou.cpu().detach().numpy() * -1)
        mbo = iou[row_ind, col_ind].mean().cpu().detach().numpy()

        return mbo

def average_precision_clevr(attr_preds, attr_gts, pos_preds, pos_gts, distance_threshold):
    B, K, num_attr = attr_gts.shape

    def process_targets(target):
        """Unpacks the target into the CLEVR properties."""
        object_size = torch.argmax(target[:2])
        material = torch.argmax(target[2:4])
        shape = torch.argmax(target[4:7])
        color = torch.argmax(target[7:15])
        real_obj = target[15]
        return object_size, material, shape, color, real_obj

    true_positives = []
    false_positives = []
    detected_gt_set = set()
    num_no_obj_pred = 0

    for b in range(B): # loops for batch
        attr_pred = attr_preds[b]
        attr_gt = attr_gts[b]
        pos_pred = pos_preds[b]
        pos_gt = pos_gts[b]

        for k_pred in range(K): # loops for pred objects
            best_distance = 10000
            best_id = None

            (pred_object_size, pred_material, pred_shape, pred_color, pred_real_obj) \
            = process_targets(attr_pred[k_pred])
            if pred_real_obj == 0:
                num_no_obj_pred += 1
                continue

            for k_gt in range(K): # loops for gt objects
                (gt_object_size, gt_material, gt_shape, gt_color, gt_real_obj) \
                = process_targets(attr_gt[k_gt])

                if gt_real_obj == 1:
                    # TODO: make the attr contents configurable 
                    attr_pred_one_obj = [pred_object_size, pred_material, pred_shape, pred_color]
                    attr_gt_one_obj = [gt_object_size, gt_material, gt_shape, gt_color]
                    match = attr_pred_one_obj == attr_gt_one_obj
                    if match:
                        distance = torch.linalg.norm(pos_pred[k_pred] - pos_gt[k_gt])
                        if distance < best_distance:
                            best_distance = distance
                            best_id = k_gt

            if (best_distance < distance_threshold) or (distance_threshold == -1):
                if best_id is not None:
                    if (b, k_pred, k_gt) not in detected_gt_set:
                        true_positives.append(1)
                        false_positives.append(0)
                        detected_gt_set.add((b, k_pred, k_gt))
                    else:
                        true_positives.append(0)
                        false_positives.append(1)
                else:
                    true_positives.append(0)
                    false_positives.append(1)
            else:
                true_positives.append(0)
                false_positives.append(1)

    assert len(true_positives) == B * K - num_no_obj_pred

    accumulated_fp = np.cumsum(np.array(false_positives))
    accumulated_tp = np.cumsum(np.array(true_positives))
    recall_array = accumulated_tp / np.sum(attr_gts.detach().cpu().numpy()[:, :, -1])
    precision_array = np.divide(accumulated_tp, (accumulated_fp + accumulated_tp))

    return compute_average_precision(
      np.array(precision_array, dtype=np.float32),
      np.array(recall_array, dtype=np.float32))

def average_precision_ptr(attr_preds, attr_gts, pos_preds, pos_gts, distance_threshold):
    B, K, num_attr = attr_gts.shape

    def process_targets(target):
        """Unpacks the target into the CLEVR properties."""
        obj_class = torch.argmax(target[:-1])
        real_obj = target[-1]
        return obj_class, real_obj

    true_positives = []
    false_positives = []
    detected_gt_set = set()
    num_no_obj_pred = 0

    for b in range(B): # loops for batch
        attr_pred = attr_preds[b]
        attr_gt = attr_gts[b]
        pos_pred = pos_preds[b]
        pos_gt = pos_gts[b]

        for k_pred in range(K): # loops for pred objects
            best_distance = 10000
            best_id = None

            pred_obj_class, pred_real_obj = process_targets(attr_pred[k_pred])
            # if pred_real_obj == 0:
            #     num_no_obj_pred += 1
            #     continue

            for k_gt in range(K): # loops for gt objects
                gt_obj_class, gt_real_obj = process_targets(attr_gt[k_gt])

                if gt_real_obj == 1:
                    # TODO: make the attr contents configurable 
                    attr_pred_one_obj = pred_obj_class
                    attr_gt_one_obj = gt_obj_class
                    match = attr_pred_one_obj == attr_gt_one_obj
                    if match:
                        distance = torch.linalg.norm(pos_pred[k_pred] - pos_gt[k_gt])
                        if distance < best_distance:
                            best_distance = distance
                            best_id = k_gt

            if (best_distance < distance_threshold) or (distance_threshold == -1):
                if best_id is not None:
                    if (b, k_gt) not in detected_gt_set:
                        true_positives.append(1)
                        false_positives.append(0)
                        detected_gt_set.add((b, k_gt))
                    else:
                        true_positives.append(0)
                        false_positives.append(1)
                else:
                    true_positives.append(0)
                    false_positives.append(1)
            else:
                true_positives.append(0)
                false_positives.append(1)

    assert len(true_positives) == B * K - num_no_obj_pred

    accumulated_fp = np.cumsum(np.array(false_positives))
    accumulated_tp = np.cumsum(np.array(true_positives))
    recall_array = accumulated_tp / np.sum(attr_gts.detach().cpu().numpy()[:, :, -1])
    precision_array = np.divide(accumulated_tp, (accumulated_fp + accumulated_tp))

    return compute_average_precision(
      np.array(precision_array, dtype=np.float32),
      np.array(recall_array, dtype=np.float32))

def compute_average_precision(precision, recall):
    """
    Code from slot-attention repo in https://github.com/google-research/google-research/
    Computation of the average precision from precision and recall arrays.
    """
    recall = recall.tolist()
    precision = precision.tolist()
    recall = [0] + recall + [1]
    precision = [0] + precision + [0]

    for i in range(len(precision) - 1, -0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    indices_recall = [
      i for i in range(len(recall) - 1) if recall[1:][i] != recall[:-1][i]
    ]

    average_precision = 0.
    for i in indices_recall:
        average_precision += precision[i + 1] * (recall[i + 1] - recall[i])
    return average_precision

