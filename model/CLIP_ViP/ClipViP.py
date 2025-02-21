import torch
import torch.nn as nn
from functools import partial
from transformers.models.clip.configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from model.CLIP_ViP.CLIP_ViP import CLIPModel, clip_loss
from model.CLIP_ViP.CLIP import CLIPModel as CLIP

class ClipVip(nn.Module):
    def __init__(self, args):
        super(ClipVip, self).__init__()
        
        # Load the CLIPViP model
        clipconfig = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32")
        
        self.clip_vision_additional_config = {
                                                "type": "ViP",
                                                "temporal_size": args.num_frames,
                                                "if_use_temporal_embed": 1,
                                                "logit_scale_init_value": 4.60,
                                                "add_cls_num": 4,
                                            }
        setattr(clipconfig, "vision_additional_config", self.clip_vision_additional_config)
        self.clipmodel = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", config=clipconfig)
        
        # init logit scale  
        logit_scale_value = 4.60
        self.clipmodel.logit_scale.data.fill_(logit_scale_value)
    
    def overload_logit_scale(self, overload_logit_scale):
        self.clipmodel.logit_scale.data.fill_(overload_logit_scale)

    def forward(self, data, image=None):
        inputs = {"input_ids": data['text']['input_ids'],
                "attention_mask": data['text']['attention_mask'],
                "pixel_values": data['video'],
                "return_loss": False}
        
        outputs = self.clipmodel(**inputs)
        text_features = outputs["text_embeds"]
        video_features = outputs["image_embeds"]

        if image:
            inputs = {"input_ids": data['caption']['input_ids'],
                    "attention_mask": data['caption']['attention_mask'],
                    "pixel_values": data['image'].unsqueeze(1),
                    "return_loss": False}
            
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

