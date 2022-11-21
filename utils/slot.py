import torch 
from collections import defaultdict

def get_slot_similarity_matrix(slots): 
    '''
    Parameters: 
        `slots`: (B, K, D_slot) 
    
    Return: 
        `similarity_matrix`: (B, K, K)
    '''

    # TODO: elaborate the similalrity computation
    slot_similarity_matrix = torch.einsum('bid, bjd -> bij', slots, slots)
    # `slot_similarity_matrix`: (B, K, K)

    return slot_similarity_matrix

def get_slot_location_id(slot_location_map, img_size, patch_size): 
    '''
    Parameters: 
        `slot_location_map`: (B, K, 2)
            -  (y, x) coordinates of slots in the input feature map
    
    Return: 
        `slot_location_ids`: (B, K)
            - `location_id`: int, 0 ~ num_patch * num_patch - 1
            - slots having the same `location_id` belong to the same local patch
    '''

    # the number of patches in a row
    n_cols = img_size // patch_size

    # assign patch_id to each slot
    slot_location_ids = n_cols * (torch.div(slot_location_map[:, :, 0], patch_size)) + torch.div(slot_location_map[:, :, 1], patch_size)
    slot_location_ids = slot_location_ids.long()
    # `slot_location_ids`: (B, K)

    return slot_location_ids

def get_slot_location_mask(slot_location_map, img_size, patch_size): 
    '''
    Parameters: 
        `slot_location_map`: (B, K, 2)

    Return: 
        `slot_location_mask`: (B, K, K)
    '''
    B, K, _ = slot_location_map.shape

    slot_location_ids = get_slot_location_id(slot_location_map, img_size, patch_size)
    # `slot_location_ids`: (B, K)

    slot_location_mask = torch.eq(slot_location_ids[:, :, None].expand(B, K, K),
                                  slot_location_ids[:, None, :].expand(B, K, K)).long()
    # `slot_location_mask`: (B, K, K)
    #   - slot_location_mask[:, i, j] = 1 or 0
    #   - indicates whether each slot belongs to the same local patch or not

    return slot_location_mask

def build_slot_pair(slots, slot_status_map, idxs, num_pairs_per_patch): 
    '''
    Parameters: 
        `slots`: (B, K, D_slot) 
        `slot_status_map`: (B, K)
            - inidicates whether a slot is activated or not
        `idxs`: (B, K*K, 2)
            - must be sorted in the decreasing order along the similarity score
        `num_pairs_per_patch`: int

    Return: 
        `slot_pairs`: (B, num_pairs, 2, D_slot)
    '''
    B, K, D_slot = slots.shape 
    N = num_pairs_per_patch
    KK = K * K
    
    slot_pairs = torch.zeros(B, N, 2, D_slot).to(slots)
    slot_pair_idxs = torch.zeros(B, N, 2).long()
    for b in range(B): 
        count = 0 # count the number of selected pairs
        check = defaultdict(int) # whether a slot is selected for merging
        for i in range(KK): 
            u, v = idxs[b, i]

            # u-th or v-th slot is already selected
            if check[u] or check[v]: 
                continue 

            # u-th or v-th slot is deactivated
            if not(slot_status_map[b, u] and slot_status_map[b, v]): 
                continue 

            # u-th and v-th slots are can be merged
            else: 
                slot_pairs[b, i, 0] = slots[b, u] 
                slot_pairs[b, i, 1] = slots[b, v]
                slot_pair_idxs[b, i, 0] = u 
                slot_pair_idxs[b, i, 1] = v

                check[u] = check[v] = 1 
                count += 1 
                if count == N: 
                    break 
        
        if count < N: 
            slot_pair_idxs[b, count:, :] = -1

    return slot_pairs, slot_pair_idxs

def update_slot_location_map(slot_location_map, img_size, patch_size):
    '''
    Parameters: 
        `slot_location_map`: (B, K, 2) 
        `img_size`: (H, W)
        `patch_size`: int 
    '''
    assert img_size % patch_size == 0, "Wrong patch size: img_size % patch_size != 0"

    return None

