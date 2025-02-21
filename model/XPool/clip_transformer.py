import torch
import torch.nn as nn
from config.base_config import Config
from model.XPool.transformer import Transformer


class XPool(nn.Module):
    def __init__(self, config: Config):
        super(XPool, self).__init__()
        self.config = config
        
        if self.config.huggingface:
            if self.config.pre_trained:
                from transformers import CLIPModel
                self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            else:
                from transformers import CLIPConfig, CLIPModel
                pretrained_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                config_clip = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32")
                self.clip = CLIPModel(config_clip)
                self.clip.text_model.load_state_dict(pretrained_clip.text_model.state_dict())
        else:
            from model.clip_model import load_clip
            self.clip = load_clip(config.clip_arch)

        config.pooling_type = 'transformer'
        self.pool_frames = Transformer(config)

        # Used for the regularized-based methods
        self.reg_params = {}
    
    def forward_video(self, video_data):
        video_data = video_data.reshape(-1, 3, self.config.input_res, self.config.input_res)
        video_features = self.clip.get_image_features(video_data)
        video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        return video_features
    
    def forward_text(self, text_input_ids, text_input_mask):
        inputs = {"input_ids": text_input_ids,
                "attention_mask": text_input_mask}
        text_features = self.clip.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def forward(self, data, return_all_frames=False, mode='train', image=None):
        batch_size = data['video'].shape[0]
        text_data = data['text']
        video_data = data['video']
        B, num_frames, C, H, W = video_data.shape

        video_data = video_data.reshape(-1, 3, self.config.input_res, self.config.input_res)
        
        if self.config.huggingface:
            text_features = self.clip.get_text_features(**text_data)
            video_features = self.clip.get_image_features(video_data)
        else:
            text_features = self.clip.encode_text(text_data)
            video_features = self.clip.encode_image(video_data)
   
        video_features = video_features.reshape(batch_size, num_frames, -1)
        # video_features = video_features.reshape(batch_size, self.config.num_frames, -1)

        if return_all_frames:
            video_features_pooled = self.pool_frames(text_features, video_features)
            if mode == 'train':
                if image:
                    image_data = data['image']
                    image_data = image_data.reshape(-1, 3, self.config.input_res, self.config.input_res)
                    if self.config.huggingface:
                        image_features = self.clip.get_image_features(image_data)
                    else:
                        image_features = self.clip.encode_image(image_data)
                    caption_data = data['caption']
                    caption_features = self.clip.get_text_features(**caption_data)
                    return text_features, video_features_pooled, image_features, caption_features
                
                return text_features, video_features_pooled
            else:
                return text_features, video_features, video_features_pooled

        return text_features, video_features
