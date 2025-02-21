import torch
import torch.nn as nn
from functools import partial
from transformers.models.clip.configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from transformers import CLIPVisionConfig
from model.FrameFusionMoE_Eval.CLIP_Fusion import CLIPModel, CLIPAttention
from model.FrameFusionMoE_Eval.adapter import LoRAAdapter

class FrameFusion(nn.Module):
    def __init__(self, args):
        super(FrameFusion, self).__init__()
        
        # Load the CLIPViP model
        clipconfig = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32")

        # Build the text and vision config
        clipconfig.text_config = self._build_text_config(args)
        clipconfig.vision_config = self._build_vision_config(args)

        self.clipmodel = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", config=clipconfig)
        
        # init logit scale  
        logit_scale_value = 4.60
        self.clipmodel.logit_scale.data.fill_(logit_scale_value)

    def overload_logit_scale(self, overload_logit_scale):
        self.clipmodel.logit_scale.data.fill_(overload_logit_scale)
    
    def _build_text_config(self, args):
        text_config = CLIPTextConfig()
        text_config.task_num = args.task_num        
        text_config.task_prototype = args.task_prototype
        text_config.lora_r = args.lora_r
        text_config.lora_alpha = args.lora_alpha
        text_config.lora_nums = args.lora_nums
        text_config.lora_dropout = args.lora_dropout
        text_config.topk = args.topk
        return text_config
    
    def _build_vision_config(self, args):
        vision_config = CLIPVisionConfig()
        vision_config.adapter_applied_layer = args.adapter_applied_layer
        return vision_config

    def reset_all_lora_counters(self):
        for module in self.modules():
            if isinstance(module, CLIPAttention):
                module.reset_lora_counters()

    def get_lora_usage_stats(self):
        choose_maps = {}
        
        # Find all LoRA adapters in model
        for name, module in self.clipmodel.named_modules():
            if isinstance(module, LoRAAdapter):
                choose_maps[name] = module.choose_map.clone()
        
        return choose_maps

    def get_gradient_stats(self):
        grad_stats = {}
        
        # Find all LoRA adapters in model
        for name, module in self.clipmodel.named_modules():
            if isinstance(module, LoRAAdapter):
                if hasattr(module, 'gradient_stats'):
                    grad_stats[name] = module.gradient_stats
                    
        return grad_stats

    def forward(self, data, image=None):
        inputs = {"input_ids": data['text']['input_ids'],
                "attention_mask": data['text']['attention_mask'],
                "pixel_values": data['video'],
                "return_loss": False}
        
        if "prototype_id" in data.keys():
            inputs["prototype_id"] = data['prototype_id']

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
    
    def forward_text(self, text_input_ids, text_input_mask, prototypes):
        inputs = {"input_ids": text_input_ids,
                "attention_mask": text_input_mask,
                "prototypes": prototypes,
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

