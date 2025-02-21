import os
import torch
from modules.basic_utils import load_json
from modules.tokenizer import clip_tokenizer
from torch.utils.data import Dataset
from config.base_config import Config
from datasets.video_capture import VideoCapture


class ACTNETDataset(Dataset):
    """
        videos_dir: directory where all videos are stored 
        config: AllConfig object
        split_type: 'train'/'test'
        img_transforms: Composition of transforms
    """
    def __init__(self, config: Config, data, model, split_type = 'train', img_transforms=None):
        self.config = config
        self.videos_train_dir = config.videos_train_dir
        self.videos_val_dir = config.videos_val_dir
        self.img_transforms = img_transforms
        self.data = data
        self.split_type = split_type

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # if model is not None:
        #     self.clip = model.to(self.device)
        #     self.clip.eval()

        from transformers import CLIPModel
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip = self.clip.to(self.device)
        self.clip.eval()

        # get all video ids for training or testing
        self.videos = [video_id for category in self.data.values() for video_id in category]

        if self.config.benchmark == 'anet_cap':
            db_file = 'datasets/ActivityNet/queries.json'
            self.vid2caption = load_json(db_file)
            if split_type == 'train':
                self._construct_all_train_pairs('anet_cap')
                self._frame_reduction(config.frame_reduction)
            else:
                self._construct_all_test_pairs('anet_cap')
        elif self.config.benchmark == 'anet_para':
            db_file = 'datasets/ActivityNet/queries.json'
            self.vid2caption = load_json(db_file)
            if split_type == 'train':
                self._construct_all_train_pairs('anet_para')
                self._frame_reduction(config.frame_reduction)
            else:
                self._construct_all_test_pairs('anet_para')  
        elif self.config.benchmark == 'anet_clip':
            if self.split_type == 'train':
                db_file = 'datasets/ActivityNet/filtered_train_queries_5.json'
                self.vid2caption = load_json(db_file)
                self._construct_all_train_pairs('anet_clip')
                self._frame_reduction(config.frame_reduction)
            else:
                db_file = 'datasets/ActivityNet/filtered_val_queries.json'
                self.vid2caption = load_json(db_file)
                self._construct_all_test_pairs('anet_clip')
        else:
            raise ValueError
    
    def __getitem__(self, index):
        if self.split_type == 'train':
            video_id, caption, frames = self._get_vidpath_and_caption_by_index_train(index)

            return {
                    'video_id': video_id,
                    'video': frames,
                    'text': caption,
            }

        else:
            video_path, caption, video_id = self._get_vidpath_and_caption_by_index_test(index)

            imgs, idxs = VideoCapture.load_frames(video_path, self.config.num_frames)

            # process images of video
            if self.img_transforms is not None:
                imgs = self.img_transforms(imgs)

            return {
                'video_id': video_id,
                'video': imgs,
                'text': caption,
            }

    def _get_vidpath_and_caption_by_index_train(self, index):
        vid, caption, frames = self.reduced_train_pairs[index]
        return vid, caption, frames

    def _get_vidpath_and_caption_by_index_test(self, index):
        vid, caption = self.all_test_pairs[index]
        video_path = os.path.join(self.videos_val_dir, vid)
        return video_path, caption, vid

    def __len__(self):
        if self.split_type == 'train':
            return len(self.all_train_pairs)
        return len(self.all_test_pairs)

    def _construct_all_train_pairs(self, benchmark):
        self.all_train_pairs = []
        for vid in self.videos:
            if benchmark == 'anet_clip':
                caption = self.vid2caption[vid]['sentences'][0]
                vid_clip = vid + '_1'
                self.all_train_pairs.append([vid_clip, caption])
            elif benchmark == 'anet_para':
                pargraph = ''
                for caption in self.vid2caption[vid]:
                    pargraph += caption + ' ' 
                self.all_train_pairs.append([vid, pargraph])
            elif benchmark == 'anet_cap':
                for caption in self.vid2caption[vid]:
                    self.all_train_pairs.append([vid, caption])

    def _construct_all_test_pairs(self, benchmark):
        self.all_test_pairs = []
        for vid in self.videos:
            if benchmark == 'anet_clip':
                caption = self.vid2caption[vid]['sentences'][0]
                vid_clip = vid + '_1'
                self.all_test_pairs.append([vid_clip, caption])
            elif benchmark == 'anet_para':
                pargraph = ''
                for caption in self.vid2caption[vid]:
                    pargraph += caption + ' ' 
                self.all_test_pairs.append([vid, pargraph])
            elif benchmark == 'anet_cap':
                for caption in self.vid2caption[vid]:
                    self.all_test_pairs.append([vid, caption])

    def _frame_reduction(self, top_k):
        print("Starting frame reduction process...")
        self.reduced_train_pairs = []

        for vid, caption in self.all_train_pairs:
            video_path = os.path.join(self.videos_train_dir, vid)
            frames, _ = VideoCapture.load_frames(video_path, self.config.num_frames)
            
            # Process frames and caption through CLIP
            if self.img_transforms:
                frames = self.img_transforms(frames)
            
            with torch.no_grad():
                frame_batch = frames.to(self.device) 
                
                # Get frame features
                frame_features = self.clip.get_image_features(frame_batch)  
                
                # Process text
                text_token = clip_tokenizer(caption, return_tensors='pt', padding=True,
                                                            truncation=True).to(self.device)
                text_features = self.clip.get_text_features(**text_token)  
                
                # Calculate similarities
                similarities = torch.nn.functional.cosine_similarity(frame_features, text_features, dim=-1)  # [T]
                
                # Get top k frames
                top_k_indices = similarities.argsort(descending=True)[:top_k].to('cpu').numpy()
                top_k_frames = frames[top_k_indices]  # Shape: [k, 3, 224, 224]
                            
            self.reduced_train_pairs.append([vid, caption, top_k_frames])

        print(f"Frame reduction complete. Reduced to {top_k} frames per video.")