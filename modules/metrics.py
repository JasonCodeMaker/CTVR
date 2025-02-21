import numpy as np
import torch
import torch.nn.functional as F
import scipy.stats
from collections import defaultdict
import os
import time

def sim_matrix_training(text_embeds, vid_embeds_pooled, pooling_type):
    """
    Computes the similarity matrix using pooled video frames
    
    Output
        sims: num_texts x num_vids
    """
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    vid_embeds_pooled = vid_embeds_pooled / vid_embeds_pooled.norm(dim=-1, keepdim=True)

    if pooling_type == 'avg':
        sims = torch.mm(text_embeds, vid_embeds_pooled.t())
        
    else:
        # num_texts x embed_dim x num_vids
        vid_embeds_pooled = vid_embeds_pooled.permute(1,2,0)
        # num_texts x 1 x embed_dim
        text_embeds = text_embeds.unsqueeze(1)
        
        sims = torch.bmm(text_embeds, vid_embeds_pooled).squeeze(1)

    return sims

def sim_matrix_training_prompt(text_embeds, vid_embeds_pooled, prompt_key, pooling_type):
    """
    Computes the similarity matrix using pooled video frames
    
    Output
        sims: num_texts x num_vids
    """
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    vid_embeds_pooled = vid_embeds_pooled / vid_embeds_pooled.norm(dim=-1, keepdim=True)

    if pooling_type == 'avg':
        sims = torch.mm(text_embeds, vid_embeds_pooled.t())
        
    else:
        # num_texts x embed_dim x num_vids
        vid_embeds_pooled = vid_embeds_pooled.permute(1,2,0)
        # num_texts x 1 x embed_dim
        text_embeds = text_embeds.unsqueeze(1)
        
        sims = torch.bmm(text_embeds, vid_embeds_pooled).squeeze(1)

        # calculate similarity between text emb and prompt key
        prompt_key = prompt_key / prompt_key.norm(dim=-1, keepdim=True)
        sims_prompt = F.cosine_similarity(prompt_key.unsqueeze(0), text_embeds.squeeze(1), dim=1)

    return sims, sims_prompt


def store_sim_matrix_inference(text_embeds_per_video_id, stored_vid_embeds_arr, mode='avg'):
    # text_embeds_per_video_id -> num_vids(curr) x max_text_per_vid x embed_dim
    # stored_vid_embeds_arr -> list of tensors, each [num_vids(stored) x num_queries x embed_dim]
    
    # Normalize text embeddings
    text_embeds_per_video_id = text_embeds_per_video_id / text_embeds_per_video_id.norm(dim=-1, keepdim=True)
    
    sims_arr = []
    for stored_vid_embeds in stored_vid_embeds_arr:
        # Normalize video embeddings
        stored_vid_embeds = stored_vid_embeds / stored_vid_embeds.norm(dim=-1, keepdim=True) # [num_vids(stored) x num_queries x embed_dim]
        # stored_vid_embeds.cuda()

        # Compute similarity
        # [num_vids(curr), max_text_per_vid, embed_dim] @ [num_vids(stored), embed_dim, num_queries]
        # -> [num_vids(curr), max_text_per_vid, num_vids(stored), num_queries]
        sims = torch.einsum('ijk,lmk->ijlm', text_embeds_per_video_id, stored_vid_embeds)

        # Average pooling over the num_queries dimension
        # [num_vids(curr), max_text_per_vid, num_vids(stored)]
        if mode == 'avg':
            sims_pooled = sims.mean(dim=-1)
        elif mode == 'max':
            sims_pooled, _ = sims.max(dim=-1)

        sims_arr.append(sims_pooled.cpu())
        # Clear CUDA cache
        del stored_vid_embeds, sims
        torch.cuda.empty_cache()
    
    # Concatenate results from all stored_vid_embeds
    return torch.cat(sims_arr, dim=-1) # [num_vids(curr), max_text_per_vid, all_num_vids(stored)]


def sim_matrix_inference(text_embeds_per_video_id, vid_embeds_pooled_per_video_id, total_num_vids, pooling_type):
    """
    Computes the similarity matrix using pooled video frames using all texts per video

    Output
        sims: num_vids x max_text_per_vid x num_vids
    """
    text_embeds_per_video_id = text_embeds_per_video_id / text_embeds_per_video_id.norm(dim=-1, keepdim=True)
    vid_embeds_pooled_per_video_id = vid_embeds_pooled_per_video_id / vid_embeds_pooled_per_video_id.norm(dim=-1, keepdim=True)

    if pooling_type == 'avg':
        # text_embeds_per_video_id -> num_vids x max_text_per_vid x embed_dim
        # vid_embeds_pooled_per_video_id -> num_vids x embed_dim

        sims = text_embeds_per_video_id @ vid_embeds_pooled_per_video_id.t()

    else:
        # text_embeds_per_video_id -> num_vids x max_text_per_vid x embed_dim
        # vid_embeds_pooled_per_video_id -> num_vids x num_vids x max_text_per_vid x embed_dim
        num_vids, max_text_per_vid, embed_dim = text_embeds_per_video_id.shape

        # num_vids x max_text_per_vid x embed_dim x num_vids
        vid_embeds_pooled_per_video_id = vid_embeds_pooled_per_video_id.permute(1,2,3,0)
        if total_num_vids is None:
            vid_embeds_pooled_per_video_id = vid_embeds_pooled_per_video_id.view(num_vids*max_text_per_vid, embed_dim, num_vids)
        else:
            vid_embeds_pooled_per_video_id = vid_embeds_pooled_per_video_id.view(num_vids*max_text_per_vid, embed_dim, total_num_vids)
        # num_vids x max_text_per_vid x 1 x embed_dim
        text_embeds_per_video_id = text_embeds_per_video_id.unsqueeze(2)
        text_embeds_per_video_id = text_embeds_per_video_id.view(num_vids*max_text_per_vid, 1, embed_dim)

        sims = torch.bmm(text_embeds_per_video_id, vid_embeds_pooled_per_video_id).cpu()
        if total_num_vids is None:
            sims = sims.view(num_vids, max_text_per_vid, num_vids).squeeze(2)
        else:
            sims = sims.view(num_vids, max_text_per_vid, 1, total_num_vids).squeeze(2)
        
    return sims

def text_embed_processing(text_embeds, all_vid_ids, max_text_per_vid):
    text_embeds_per_video_id = defaultdict(list)

    for embed, vid in zip(text_embeds, all_vid_ids):
        text_embeds_per_video_id[vid].append(embed)

    text_embeds_per_video_id = {vid: torch.stack(embs) for vid, embs in text_embeds_per_video_id.items()}

    # num_vids x max_text_per_vid x embed_dim
    text_embeds_per_video_id = pad_and_stack_dict_to_tensor(text_embeds_per_video_id,
            text_embeds_per_video_id.keys(), text_embeds.shape[-1], max_text_per_vid)
    
    return text_embeds_per_video_id

def vid_embed_pooled_processing(vid_embeds_pooled, all_vid_ids, max_text_per_vid):
    num_vids = vid_embeds_pooled.shape[0]
    vid_embeds_pooled_per_video_id = []

    for i in range(num_vids):
        vid_dict = defaultdict(list)

        for embed, vid in zip(vid_embeds_pooled[i], all_vid_ids):
            vid_dict[vid].append(embed)
        
        # Stack tensors for each video ID
        vid_dict = {vid: torch.stack(embs) for vid, embs in vid_dict.items()}

        # num_vids x max_text_per_vid x embed_dim
        vid_embeds_pooled_per_video_id.append(
            pad_and_stack_dict_to_tensor(
                vid_dict, 
                vid_dict.keys(), 
                vid_embeds_pooled.shape[-1],
                max_text_per_vid
            )
        )

        # Clear CUDA cache
        torch.cuda.empty_cache()
    
    # num_vids x num_vids x max_text_per_vid x embed_dim
    return torch.stack(vid_embeds_pooled_per_video_id)

def generate_embeds_per_video_id(text_embeds, vid_embeds_pooled, all_vid_ids, max_text_per_vid, pooling_type):
    # Construct dictionary of text embeds per unique video id
    text_embeds_per_video_id = text_embed_processing(text_embeds, all_vid_ids, max_text_per_vid)

    if pooling_type == 'avg':
        # num_vids x embed_dim
        vid_embeds_pooled_per_video_id = vid_embeds_pooled

    else:
        # Construct dictionary of video embeds for each text per video_id
        vid_embeds_pooled_per_video_id = vid_embed_pooled_processing(vid_embeds_pooled, all_vid_ids, max_text_per_vid)

    return text_embeds_per_video_id, vid_embeds_pooled_per_video_id

def t2v_metrics(sims):
    # Permute sims so it represents a sequence of text-video similarity matrices.
    # Then obtain the double argsort to position the rank on the diagonal
    stacked_sims = sims.permute(1, 0, 2)
    
    sims_sort = torch.argsort(stacked_sims, dim=-1, descending=True)
    sims_sort_2 = torch.argsort(sims_sort, dim=-1, descending=False)

    ranks = torch.flatten(torch.diagonal(sims_sort_2, dim1=1, dim2=2))
    
    # Now we need to extract valid ranks, as some belong to inf padding values
    valid_check = torch.flatten(torch.diagonal(sims, dim1=0, dim2=2))
    mask = ~torch.logical_or(torch.isinf(valid_check), torch.isnan(valid_check))
    valid_ranks = ranks[mask]

    return compute_metrics(valid_ranks.cpu().numpy())

def t2v_eval_metrics_cl(sims, vid_start_idx, vid_end_idx, num_queries):
    stacked_sims = sims.permute(1, 0, 2)  # [1, num_queries, total_videos]
    
    sims_sort = torch.argsort(stacked_sims, dim=-1, descending=True)
    sims_sort_2 = torch.argsort(sims_sort, dim=-1, descending=False)
    
    valid_ranks = []
    for i in range(num_queries):
        if vid_start_idx + i < vid_end_idx:  
            gt_vid_idx = vid_start_idx + i 
            rank = sims_sort_2[0, i, gt_vid_idx]
            valid_ranks.append(rank)
            
    valid_ranks = torch.tensor(valid_ranks).cpu().numpy()

    metrics = {}
    metrics["R1"] = 100 * float(np.sum(valid_ranks == 0)) / len(valid_ranks)
    metrics["R5"] = 100 * float(np.sum(valid_ranks < 5)) / len(valid_ranks)
    metrics["R10"] = 100 * float(np.sum(valid_ranks < 10)) / len(valid_ranks)
    
    return metrics

def v2t_metrics(sims):
    # Code to avoid nans
    sims[sims!=sims] = float('-inf')
    # Forms a similarity matrix
    sims, _ = torch.max(sims, dim = 1)
    sims = sims.t()

    sims_sort = torch.argsort(sims, dim=-1, descending=True)
    sims_sort_2 = torch.argsort(sims_sort, dim=-1, descending=False)

    ranks = torch.diag(sims_sort_2).numpy() # diagonal

    return compute_metrics(ranks)

def compute_metrics(lst):
    metrics = {}
    metrics["R1"] = 100 * float(np.sum(lst == 0)) / len(lst)
    metrics["R5"] = 100 * float(np.sum(lst < 5)) / len(lst)
    metrics["R10"] = 100 * float(np.sum(lst < 10)) / len(lst)
    metrics["R50"] = 100 * float(np.sum(lst < 50)) / len(lst)
    metrics["R100"] = 100 * float(np.sum(lst < 100)) / len(lst)
    # metrics["sum_ranks"] = np.sum(lst)
    # metrics["all_ranks"] = lst.tolist()
    metrics["MedR"] = np.median(lst) + 1
    metrics["MeanR"] = np.mean(lst) + 1
    #stats = [metrics[x] for x in ("R1", "R5", "R10")]
    #metrics["geometric_mean_R1-R5-R10"] = scipy.stats.mstats.gmean(stats)
    return metrics


def pad_and_stack_dict_to_tensor(input, order, d, max_length):
    if max_length is None:
        max_length = max([input[k].shape[0] for k in input])
    
    padded_input = {k: torch.cat([input[k], torch.full((max_length - input[k].shape[0], d), 
                                                        float("-inf"), device = input[k].device)]) for k in input}
    
    padded_stacked_input = torch.stack([padded_input[k] for k in order], dim = 0)
    return padded_stacked_input
