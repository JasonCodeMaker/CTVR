import torch.nn as nn
import torch
import torch.nn.functional as F

from .tokenizer import clip_tokenizer

class CLIPPromptLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sims, logit_scale, prompt_sim):
        """
        Inputs: cosine similarities
            sims: n x n (text is dim-0)
            logit_scale: 1 x 1
        """
        logit_scale = logit_scale.exp()
        logits = sims * logit_scale
        
        t2v_log_sm = F.log_softmax(logits, dim=1)
        t2v_neg_ce = torch.diag(t2v_log_sm)
        t2v_loss = -t2v_neg_ce.mean()

        v2t_log_sm = F.log_softmax(logits, dim=0)
        v2t_neg_ce = torch.diag(v2t_log_sm)
        v2t_loss = -v2t_neg_ce.mean()

        prompt_log_sm = F.log_softmax(prompt_sim, dim=0)
        prompt_loss = -prompt_log_sm.mean()

        return (t2v_loss + v2t_loss) / 2.0 + 1.0 * prompt_loss

# X-Pool
class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sims, logit_scale):
        """
        Inputs: cosine similarities
            sims: n x n (text is dim-0)
            logit_scale: 1 x 1
        """
        logit_scale = logit_scale.exp()
        logits = sims * logit_scale
        
        t2v_log_sm = F.log_softmax(logits, dim=1)
        t2v_neg_ce = torch.diag(t2v_log_sm)
        t2v_loss = -t2v_neg_ce.mean()

        v2t_log_sm = F.log_softmax(logits, dim=0)
        v2t_neg_ce = torch.diag(v2t_log_sm)
        v2t_loss = -v2t_neg_ce.mean()

        return (t2v_loss + v2t_loss) / 2.0 

# Switch Prompt
class CaptionLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        weight = torch.ones(clip_tokenizer.vocab_size)
        frequent_words = "in on of to about this that a an the and there here is are . , \
                          <|endoftext|> <|startoftext|>"
        frequent_ids = clip_tokenizer.convert_tokens_to_ids(
            clip_tokenizer.tokenize(frequent_words)
        )
        weight[frequent_ids] = config.frequent_word_weight
        self.register_buffer('weight', weight)
        self.mult = config.caption_loss_mult

    def forward(self, pred_logits, input_ids):
        mask = input_ids[:, :-1] != clip_tokenizer.eos_token_id
        pred_logits = pred_logits[mask]
        target_ids = input_ids[:, 1:][mask]
        return F.cross_entropy(pred_logits, 
                               target_ids, 
                               weight=self.weight) * self.mult

class NCELearnableTempLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self):
        super(NCELearnableTempLoss, self).__init__()

    def forward(self, vis_feat, text_feat, temp):
        logit_scale = temp.exp()
        t2v = torch.matmul(vis_feat, text_feat.permute(1, 0)) * logit_scale  # temperature
        v2t = t2v.permute(1, 0)
        t2v_label = torch.arange(t2v.shape[0], device=t2v.device)
        v2t_label = t2v_label
        loss = (F.cross_entropy(t2v, t2v_label) + F.cross_entropy(v2t, v2t_label)).mean()
        return loss

class TripletLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self):
        super(TripletLoss, self).__init__()

    # def class_wise_contrastive_loss(self, vis_feat, text_feat, categories, temp, margin=0.5):

    #     # Convert categories to tensor of indices
    #     unique_categories = list(set(categories))
    #     category_to_idx = {cat: idx for idx, cat in enumerate(unique_categories)}
    #     vis_labels = torch.tensor([category_to_idx[cat] for cat in categories], 
    #                             device=vis_feat.device)
    #     text_labels = vis_labels  # Since they share the same categories
        
    #     # Calculate similarity matrix
    #     logit_scale = temp.exp()
    #     sim_matrix = torch.matmul(vis_feat, text_feat.t()) * logit_scale
        
    #     # Create label matrix
    #     label_matrix = (vis_labels.unsqueeze(1) == text_labels.unsqueeze(0)).float()
        
    #     # Compute positive and negative losses
    #     pos_loss = (-sim_matrix * label_matrix).sum() / (label_matrix.sum() + 1e-6)
    #     neg_loss = (torch.clamp(sim_matrix - margin, min=0.0) * (1 - label_matrix)).sum() / ((1 - label_matrix).sum() + 1e-6)
        
    #     return pos_loss + neg_loss

    def forward(self, vis_feat, text_feat, temp, ref_vis_feat, scale=0.5, category=None):
        logit_scale = temp.exp()
        t2v = torch.matmul(vis_feat, text_feat.permute(1, 0)) * logit_scale  # temperature
        v2t = t2v.permute(1, 0)
        t2v_label = torch.arange(t2v.shape[0], device=t2v.device)
        v2t_label = t2v_label
        NCE_loss = (F.cross_entropy(t2v, t2v_label) + F.cross_entropy(v2t, v2t_label)).mean()

        # Triplet loss
        if ref_vis_feat is not None:
            # Calculate the similarity between the text and the positive video
            txt_img_sim = torch.matmul(text_feat, vis_feat.permute(1, 0)) * logit_scale
            txt_neg_sim = torch.matmul(text_feat, ref_vis_feat.permute(1, 0)) * logit_scale
            labels = torch.arange(txt_img_sim.shape[0], device=txt_img_sim.device)

            # Calculate the triplet loss
            triplet_loss = F.cross_entropy(torch.cat([txt_img_sim, txt_neg_sim], dim=-1), labels)
            loss = (1-scale) * NCE_loss + scale * triplet_loss
        else:
            loss = NCE_loss

        return loss

def distillation(t, s, T=2.0):
    p = F.softmax(t / T, dim=1)
    loss = F.cross_entropy(s / T, p, reduction="mean") * (T ** 2)
    return loss

class NCELearnableTempLoss_lwf(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self):
        super(NCELearnableTempLoss_lwf, self).__init__()

    def forward(self, vis_feat, text_feat, temp, ref_vis_feat, ref_text_feat):
        logit_scale = temp.exp()
        t2v = torch.matmul(vis_feat, text_feat.permute(1, 0)) * logit_scale  # temperature
        v2t = t2v.permute(1, 0)
        t2v_label = torch.arange(t2v.shape[0], device=t2v.device)
        v2t_label = t2v_label
        contrastive_loss = (F.cross_entropy(t2v, t2v_label) + F.cross_entropy(v2t, v2t_label)).mean()

        # LwF loss
        curr_t2v = torch.matmul(vis_feat, ref_text_feat.permute(1, 0)) * logit_scale
        curr_v2t = curr_t2v.permute(1, 0)
        ref_t2v = torch.matmul(ref_vis_feat, ref_text_feat.permute(1, 0)) * logit_scale  # temperature
        ref_v2t = ref_t2v.permute(1, 0)
        distill_loss = (distillation(ref_t2v, curr_t2v) + distillation(ref_v2t, curr_v2t)).mean()

        loss = contrastive_loss + distill_loss

        return loss

class NCELearnableTempLoss_zscl(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self):
        super(NCELearnableTempLoss_zscl, self).__init__()

    def forward(self, vis_feat, text_feat, temp, ref_vis_feat_curr, ref_vis_feat, ref_text_feat):
        logit_scale = temp.exp()
        t2v = torch.matmul(vis_feat, text_feat.permute(1, 0)) * logit_scale  # temperature
        v2t = t2v.permute(1, 0)
        t2v_label = torch.arange(t2v.shape[0], device=t2v.device)
        v2t_label = t2v_label
        contrastive_loss = (F.cross_entropy(t2v, t2v_label) + F.cross_entropy(v2t, v2t_label)).mean()

        # ZSCL loss
        curr_t2v = torch.matmul(ref_vis_feat_curr, ref_text_feat.permute(1, 0)) * logit_scale
        curr_v2t = curr_t2v.permute(1, 0)
        ref_t2v = torch.matmul(ref_vis_feat, ref_text_feat.permute(1, 0)) * logit_scale  # temperature
        ref_v2t = ref_t2v.permute(1, 0)
        distill_loss = (distillation(ref_t2v, curr_t2v) + distillation(ref_v2t, curr_v2t)).mean()

        loss = contrastive_loss + distill_loss

        return loss

class NCELearnableTempLoss_moe(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self):
        super(NCELearnableTempLoss_moe, self).__init__()

    def forward(self, vis_feat, text_feat, temp, router_weight, prev_router_weight, scale):

        # def print_grad(grad):
        #     print("Gradient flowing through router_weight:", grad)
        
        # router_weight.register_hook(print_grad)

        logit_scale = temp.exp()
        t2v = torch.matmul(vis_feat, text_feat.permute(1, 0)) * logit_scale
        v2t = t2v.permute(1, 0)
        t2v_label = torch.arange(t2v.shape[0], device=t2v.device)
        v2t_label = t2v_label
        contrastive_loss = (F.cross_entropy(t2v, t2v_label) + F.cross_entropy(v2t, v2t_label)).mean()
        
        # Check MoE loss gradients
        moe_loss = torch.tensor(0.0, device=contrastive_loss.device, dtype=contrastive_loss.dtype)

        if len(prev_router_weight) == 0:
            return contrastive_loss, contrastive_loss, moe_loss
        else:
            num_tasks = len(prev_router_weight)
            
            for i, e_old in enumerate(prev_router_weight):
                # Compute similarity for this task
                similarity = (router_weight * e_old).sum(dim=-1).sum()
                moe_loss += similarity

            log_router_weight = F.log_softmax(router_weight, dim=-1)
            for i, e_old in enumerate(prev_router_weight):
                prev_prob = F.softmax(e_old, dim=-1)
                kl_div = F.kl_div(log_router_weight, prev_prob, reduction='sum')
                moe_loss -= kl_div

            moe_loss /= num_tasks
            
            # Final loss
            scale_tensor = torch.tensor(scale, device=moe_loss.device, dtype=moe_loss.dtype)
            total_loss = contrastive_loss + scale_tensor * moe_loss

            return total_loss, contrastive_loss, scale_tensor * moe_loss

class NCELearnableTempLoss_vt_ft(nn.Module):
    """
    Compute contrastive loss: video-(sub,cap)
    """

    def __init__(self):
        super(NCELearnableTempLoss_vt_ft, self).__init__()

    def forward(self, vis_feat, text_feat, img_feat, caption_feat, temp):
        logit_scale = temp.exp()
        # V-T
        t2v = torch.matmul(vis_feat, text_feat.permute(1, 0)) * logit_scale  # temperature
        v2t = t2v.permute(1, 0)
        t2v_label = torch.arange(t2v.shape[0], device=t2v.device)
        v2t_label = t2v_label
        # F-C
        v2t_3 = torch.matmul(img_feat, caption_feat.permute(1, 0)) * logit_scale  # temperature
        t2v_3 = v2t_3.permute(1, 0)
        t2v_label_3 = torch.arange(t2v_3.shape[0], device=t2v_3.device)
        v2t_label_3 = t2v_label_3

        loss = (F.cross_entropy(t2v, t2v_label) + F.cross_entropy(v2t, v2t_label) + \
            F.cross_entropy(t2v_3, t2v_label_3) + F.cross_entropy(v2t_3, v2t_label_3)).mean()
        
        img_loss = (F.cross_entropy(t2v_3, t2v_label_3) + F.cross_entropy(v2t_3, v2t_label_3)).mean()

        return loss, img_loss

class CLIPLoss_vt_ft(nn.Module):
    """
    Compute contrastive loss: video-(sub,cap)
    """

    def __init__(self):
        super(CLIPLoss_vt_ft, self).__init__()

    def forward(self, vis_feat, text_feat, img_feat, caption_feat, temp):
        logit_scale = temp.exp()
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        vis_feat = vis_feat / vis_feat.norm(dim=-1, keepdim=True)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        caption_feat = caption_feat / caption_feat.norm(dim=-1, keepdim=True)

        # V-T
        vis_feat_pooled = vis_feat.permute(1,2,0)
        text_feat = text_feat.unsqueeze(1)
        t2v = torch.bmm(text_feat, vis_feat_pooled).squeeze(1) * logit_scale  # temperature
        t2v_log_sm = F.log_softmax(t2v, dim=1)
        t2v_neg_ce = torch.diag(t2v_log_sm)
        t2v_loss = -t2v_neg_ce.mean()

        v2t_log_sm = F.log_softmax(t2v, dim=0)
        v2t_neg_ce = torch.diag(v2t_log_sm)
        v2t_loss = -v2t_neg_ce.mean()

        # F-C
        v2t_3 = torch.mm(caption_feat, img_feat.t()) * logit_scale  # temperature
        v2t_log_sm_3 = F.log_softmax(v2t_3, dim=0)
        v2t_neg_ce_3 = torch.diag(v2t_log_sm_3)
        v2t_loss_3 = -v2t_neg_ce_3.mean()

        t2v_log_sm_3 = F.log_softmax(v2t_3, dim=1)
        t2v_neg_ce_3 = torch.diag(t2v_log_sm_3)
        t2v_loss_3 = -t2v_neg_ce_3.mean()

        loss = (t2v_loss + v2t_loss + t2v_loss_3 + v2t_loss_3) / 4.0
        img_loss = (t2v_loss_3 + v2t_loss_3) / 2.0
        return loss, img_loss

class LossFactory:
    @staticmethod
    def get_loss(config):
        if config.loss == 'clip':
            return CLIPLoss()
        elif config.loss == 'clip_vt_ft':
            return CLIPLoss_vt_ft()
        elif config.loss == 'clip_prompt':
            return CLIPPromptLoss()
        elif config.loss == 'NCELearnableTempLoss':
            return NCELearnableTempLoss()
        elif config.loss == 'lwf':
            return NCELearnableTempLoss_lwf()
        elif config.loss == 'zscl':
            return NCELearnableTempLoss_zscl()
        elif config.loss == 'triplet':
            return TripletLoss()
        elif config.loss == 'NCELearnableTempLoss_moe':
            return NCELearnableTempLoss_moe()
        elif config.loss == 'NCELearnableTempLoss_vt_ft':
            return NCELearnableTempLoss_vt_ft()
        elif config.loss == 'clip+caption':
            return {'clip': CLIPLoss(),
                    'caption': CaptionLoss(config)}
        else:
            raise NotImplemented
