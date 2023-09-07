import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
from copy import deepcopy
from typing import Dict, List

from .revisited import compute_metrics, general_compute_map
from .data.utils import json_save, pickle_save

from torch.multiprocessing import Pool, Process, set_start_method

import time

import sys

torch.set_num_threads(1)

class AverageMeter:
    """Computes and stores the average and current value on device"""

    def __init__(self, device, length):
        self.device = device
        self.length = length
        self.reset()

    def reset(self):
        self.values = torch.zeros(self.length, device=self.device, dtype=torch.float)
        self.counter = 0
        self.last_counter = 0

    def append(self, val):
        self.values[self.counter] = val.detach()
        self.counter += 1
        self.last_counter += 1

    @property
    def val(self):
        return self.values[self.counter - 1]

    @property
    def avg(self):
        return self.values[:self.counter].mean()

    @property
    def values_list(self):
        return self.values[:self.counter].cpu().tolist()

    @property
    def last_avg(self):
        if self.last_counter == 0:
            return self.latest_avg
        else:
            self.latest_avg = self.values[self.counter - self.last_counter:self.counter].mean()
            self.last_counter = 0
            return self.latest_avg


@torch.no_grad()
def calculate_score(i, model, 
            query_global, query_local, query_mask, query_scales, query_positions,
            index_global, index_local,index_mask, index_scales, index_positions):

    print("Computing for " + str(i))
    
    current_scores = model(query_global, query_local, query_mask, query_scales, query_positions,
            index_global, index_local,index_mask, index_scales, index_positions)
    
    print("Computer score:  " + str(current_scores.cpu().data))
    return ((i, current_scores.cpu().data))

@torch.no_grad()
def mean_average_precision_revisited_rerank(
    model: nn.Module,
    cache_nn_inds: torch.Tensor,
    query_global: torch.Tensor, query_local: torch.Tensor, query_mask: torch.Tensor, query_scales: torch.Tensor, query_positions: torch.Tensor,
    gallery_global: torch.Tensor, gallery_local: torch.Tensor, gallery_mask: torch.Tensor, gallery_scales: torch.Tensor, gallery_positions: torch.Tensor,
    ks: List[int],
    gnd, train = False, limit = 100) -> Dict[str, float]:

    # orig_device = next(model.parameters()).device
    # orig_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = next(model.parameters()).device
    # device = torch.device('cpu')
    # model.to(device)
    
    query_global    = query_global.to(device)
    query_local     = query_local.to(device)
    query_mask      = query_mask.to(device)
    query_scales    = query_scales.to(device)
    query_positions = query_positions.to(device)

    num_samples, top_k = cache_nn_inds.size()
    
    if limit is not None: 
        top_k = min(limit, top_k)

    ########################################################################################
    ## Medium
    medium_nn_inds = deepcopy(cache_nn_inds.cpu().data.numpy())

    # Exclude the junk images as in DELG (https://github.com/tensorflow/models/blob/44cad43aadff9dd12b00d4526830f7ea0796c047/research/delf/delf/python/detect_to_retrieve/image_reranking.py#L190)
    
    if gnd is not None:
        for i in range(num_samples):
            junk_ids = gnd['gnd'][i]['junk']
            all_ids = medium_nn_inds[i]
            pos = np.in1d(all_ids, junk_ids)
            neg = np.array([not x for x in pos])
            new_ids = np.concatenate([np.arange(len(all_ids))[neg], np.arange(len(all_ids))[pos]])
            new_ids = all_ids[new_ids]
            medium_nn_inds[i] = new_ids
    else: 
        for i in range(num_samples):
            junk_ids = []
            all_ids = medium_nn_inds[i]
            pos = np.in1d(all_ids, junk_ids)
            neg = np.array([not x for x in pos])

            new_ids = np.concatenate([np.arange(len(all_ids))[neg], np.arange(len(all_ids))[pos]])
            new_ids = all_ids[new_ids]
            medium_nn_inds[i] = new_ids
    
    medium_nn_inds = torch.from_numpy(medium_nn_inds)
    print(medium_nn_inds)

    scores = []
    # scores = [-1] * top_k
    # inputs = []

    # for i in tqdm(range(top_k), desc='Computing inputs'):
    #     nnids = medium_nn_inds[:, i]
    #     index_global    = gallery_global[nnids].to(device)

    #     index_local     = gallery_local[nnids].to(device)
    #     index_mask      = gallery_mask[nnids].to(device)
    #     index_scales    = gallery_scales[nnids].to(device)
    #     index_positions = gallery_positions[nnids].to(device)

    #     inputs.append((i, model, 
    #         query_global, query_local, query_mask, query_scales, query_positions,
    #         index_global, index_local,index_mask, index_scales, index_positions))
           

    # print("Defined calculate_score_function")
    # try:
    #  set_start_method('spawn', force=True)
    # except RuntimeError:
    #     print("Runtime Error")
    #     pass
    
    # print("Creating Pool: ")
    # multi_pool = Pool(processes=top_k)
    # results = multi_pool.starmap(calculate_score, inputs)
    # multi_pool.close() 
    # multi_pool.join()

    # print(results)

    # for elem in results:
    #     scores[elem[0]] = elem[1]

    for i in tqdm(range(top_k), desc='Computing current scores'):
        nnids = medium_nn_inds[:, i]

        index_global    = gallery_global[nnids]
        index_local     = gallery_local[nnids]
        index_mask      = gallery_mask[nnids]
        index_scales    = gallery_scales[nnids]
        index_positions = gallery_positions[nnids]

        current_scores = model(
            query_global, query_local, query_mask, query_scales, query_positions,
            index_global.to(device), 
            index_local.to(device),
            index_mask.to(device), 
            index_scales.to(device), 
            index_positions.to(device))
        
        scores.append(current_scores.cpu().data)
    
    scores = torch.stack(scores, -1) # 70 x 100
    closest_dists, indices = torch.sort(scores, dim=-1, descending=True)
    closest_indices = torch.gather(medium_nn_inds, -1, indices)
    ranks = deepcopy(medium_nn_inds)
    ranks[:, :top_k] = deepcopy(closest_indices)
    ranks = ranks.cpu().data.numpy()

    # model.to(orig_device)

    if not train: 
        return ranks
    else:
        return general_compute_map(ranks, dataset_location="/scratch/as216/amur/reid_synthetic_train.csv")

    # pickle_save('medium_nn_inds.pkl', ranks.T)

    if gnd is not None:
        medium = compute_metrics('revisited', ranks, gnd['gnd'], kappas=ks)
    else: 
        medium = compute_metrics('revisited', ranks, None, kappas=ks)

    ########################################################################################
    ## Hard
    hard_nn_inds = deepcopy(cache_nn_inds.cpu().data.numpy())
    # Exclude the junk images as in DELG (https://github.com/tensorflow/models/blob/44cad43aadff9dd12b00d4526830f7ea0796c047/research/delf/delf/python/detect_to_retrieve/image_reranking.py#L190)
    
    if gnd is not None:
        for i in range(num_samples):
            junk_ids = gnd['gnd'][i]['junk'] + gnd['gnd'][i]['easy']
            all_ids = hard_nn_inds[i]
            pos = np.in1d(all_ids, junk_ids)
            neg = np.array([not x for x in pos])
            new_ids = np.concatenate([np.arange(len(all_ids))[neg], np.arange(len(all_ids))[pos]])
            new_ids = all_ids[new_ids]
            hard_nn_inds[i] = new_ids

    hard_nn_inds = torch.from_numpy(hard_nn_inds)

    scores = []
    for i in tqdm(range(top_k)):
        nnids = hard_nn_inds[:, i]
        index_global    = gallery_global[nnids]
        index_local     = gallery_local[nnids]
        index_mask      = gallery_mask[nnids]
        index_scales    = gallery_scales[nnids]
        index_positions = gallery_positions[nnids]
        current_scores = model(
            query_global, query_local, query_mask, query_scales, query_positions,
            index_global.to(device),
            index_local.to(device),
            index_mask.to(device),
            index_scales.to(device),
            index_positions.to(device))
        scores.append(current_scores.cpu().data)
    scores = torch.stack(scores, -1) # 70 x 100
    closest_dists, indices = torch.sort(scores, dim=-1, descending=True)
    closest_indices = torch.gather(hard_nn_inds, -1, indices)

    # pickle_save('nn_inds_rerank.pkl', closest_indices)
    # pickle_save('nn_dists_rerank.pkl', closest_dists)

    ranks = deepcopy(hard_nn_inds)
    ranks[:, :top_k] = deepcopy(closest_indices)
    ranks = ranks.cpu().data.numpy().T
    # pickle_save('hard_nn_inds.pkl', ranks.T)

    if gnd is not None:
        hard = compute_metrics('revisited', ranks, None, kappas=ks)
    else: 
        hard = compute_metrics('revisited', ranks, gnd['gnd'], kappas=ks)

    ########################################################################################  
    out = {
        'M_map': float(medium['M_map']), 
        'H_map': float(hard['H_map']),
        'M_mp':  medium['M_mp'].tolist(),
        'H_mp': hard['H_mp'].tolist(),
    }
    # json_save('eval_revisited.json', out)
    return out


@torch.no_grad()
def mean_average_precision_revisited_rerank_time(
    model: nn.Module,
    cache_nn_inds: torch.Tensor,
    query_global: torch.Tensor, query_local: torch.Tensor, query_mask: torch.Tensor, query_scales: torch.Tensor, query_positions: torch.Tensor,
    gallery_global: torch.Tensor, gallery_local: torch.Tensor, gallery_mask: torch.Tensor, gallery_scales: torch.Tensor, gallery_positions: torch.Tensor,
    ks: List[int],
    gnd) -> Dict[str, float]:

    orig_device = next(model.parameters()).device
    new_device = torch.device('cpu')
    model = model.to(device)

    # query_global    = query_global.to(device)
    # query_local     = query_local.to(device)
    # query_mask      = query_mask.to(device)
    # query_scales    = query_scales.to(device)
    # query_positions = query_positions.to(device)

    num_samples, top_k = cache_nn_inds.size()
    top_k = min(10, top_k)

    ########################################################################################
    ## Medium
    medium_nn_inds = deepcopy(cache_nn_inds.cpu().data.numpy())
    for i in range(num_samples):
        junk_ids = gnd['gnd'][i]['junk']
        all_ids = medium_nn_inds[i]
        pos = np.in1d(all_ids, junk_ids)
        neg = np.array([not x for x in pos])
        new_ids = np.concatenate([np.arange(len(all_ids))[neg], np.arange(len(all_ids))[pos]])
        new_ids = all_ids[new_ids]
        medium_nn_inds[i] = new_ids
    medium_nn_inds = torch.from_numpy(medium_nn_inds)
    
    scores = []
    total_time = 0.0
    for i in range(num_samples):
        nnids = medium_nn_inds[i,:top_k]
        index_global    = gallery_global[nnids]
        index_local     = gallery_local[nnids]
        index_mask      = gallery_mask[nnids]
        index_scales    = gallery_scales[nnids]
        index_positions = gallery_positions[nnids]


        src_global    = query_global[i].unsqueeze(0).repeat(top_k, 1)
        src_local     = query_local[i].unsqueeze(0).repeat(top_k, 1, 1)
        src_mask      = query_mask[i].unsqueeze(0).repeat(top_k, 1)
        src_scales    = query_scales[i].unsqueeze(0).repeat(top_k, 1)
        src_positions = query_positions[i].unsqueeze(0).repeat(top_k, 1, 1)

        index_global = index_global.to(device)
        index_local = index_local.to(device)
        index_mask = index_mask.to(device)
        index_scales = index_scales.to(device)
        index_positions = index_positions.to(device)

        start = time.time()
        current_scores = model(
            src_global, src_local, src_mask, src_scales, src_positions,
            index_global,
            index_local,
            index_mask,
            index_scales,
            index_positions)
        end = time.time()
        total_time += end-start

        scores.append(current_scores.cpu().data)
    scores = torch.stack(scores, 0) # 70 x 100
    print('scores', scores.shape)
    print('time', total_time/num_samples)
    closest_dists, indices = torch.sort(scores, dim=-1, descending=True)
    closest_indices = torch.gather(medium_nn_inds, -1, indices)
    ranks = deepcopy(medium_nn_inds)
    ranks[:, :top_k] = deepcopy(closest_indices)
    ranks = ranks.cpu().data.numpy().T
    medium = compute_metrics('revisited', ranks, gnd['gnd'], kappas=ks)

    ########################################################################################  
    out = {
        'M_map': float(medium['M_map']), 
        'M_mp':  medium['M_mp'].tolist(),
    }
    # json_save('eval_revisited.json', out)

    model.to(orig_device)
    return out