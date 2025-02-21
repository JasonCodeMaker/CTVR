import torch
import torch.nn as nn
from functools import partial
from transformers.models.clip.configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from transformers import CLIPVisionConfig
from model.MoE_Adapter.MoECLIP import CLIPModel, clip_loss

class MoEAdapter(nn.Module):
    def __init__(self, args):
        super(MoEAdapter, self).__init__()  
        # Load the CLIPViP model
        clipconfig = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32")
        clipconfig.text_config = CLIPTextConfig()
        clipconfig.vision_config = CLIPVisionConfig()

        self.clip_moe_config = {
                                    "task_id": args.task_id, # router id
                                    "ffn_num": args.ffn_num, # ffn hidden size
                                    "ffn_adapt": args.ffn_adapt, # expert
                                    "ffn_option": args.ffn_option, # "parallel"
                                    "ffn_adapt_where": args.ffn_adapt_where, # both Image and Text
                                    "apply_moe": args.apply_moe , # True
                                    "experts_num": args.num_experts, # 22
                                    "is_train": True, # Train or Val
                                    "task_num": args.task_num, # 10
                                    "topk": args.topk, # 2
                                    "autorouter": True, # Use autoencoder or not
                                }
        setattr(clipconfig, "moe_config", self.clip_moe_config)
        self.clipmodel = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", config=clipconfig)
        
        # init logit scale  
        logit_scale_value = 4.60
        self.clipmodel.logit_scale.data.fill_(logit_scale_value)
    
    def overload_logit_scale(self, overload_logit_scale):
        self.clipmodel.logit_scale.data.fill_(overload_logit_scale)

    def train(self, mode=True):
        self.is_train = mode
        for module in self.children():
            if isinstance(module, MoEAdapter):
                module.train(mode)
            elif isinstance(module, nn.Module):
                module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def forward(self, data, image=None, task_id=None):
        inputs = {"input_ids": data['text']['input_ids'],
                "attention_mask": data['text']['attention_mask'],
                "pixel_values": data['video'],
                "return_loss": False,
                "task_id": task_id}

        outputs = self.clipmodel(**inputs)
        text_features = outputs["text_embeds"]
        video_features = outputs["image_embeds"]

        if image:
            inputs = {"input_ids": data['caption']['input_ids'],
                    "attention_mask": data['caption']['attention_mask'],
                    "pixel_values": data['image'].unsqueeze(1),
                    "return_loss": False,
                    "task_id": task_id}
            
            outputs = self.clipmodel(**inputs)
            caption_features = outputs["text_embeds"]
            image_features = outputs["image_embeds"]

            return text_features, video_features, image_features, caption_features

        return text_features, video_features
    
    def forward_video(self, video):
        inputs = {"pixel_values": video,
                "if_norm": True}
        video_features = self.clipmodel.get_image_features(**inputs)
        return video_features
    
    def forward_text(self, text_input_ids, text_input_mask):
        inputs = {"input_ids": text_input_ids,
                "attention_mask": text_input_mask,
                "if_norm": True}
        text_features = self.clipmodel.get_text_features(**inputs)
        return text_features

    def freeze_text_encoder(self, freeze_text_proj):
        freeze_list = [self.clipmodel.text_model]
        if freeze_text_proj:
            freeze_list.append(self.clipmodel.text_projection)
        for m in freeze_list:
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

