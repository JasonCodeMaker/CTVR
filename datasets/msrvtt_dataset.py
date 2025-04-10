import os
import torch
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from modules.basic_utils import load_json
from modules.tokenizer import clip_tokenizer
from torch.utils.data import Dataset
from datasets.video_capture import VideoCapture
from transformers import CLIPModel
from datasets.utils.model_transforms import init_transform_dict
from datetime import timedelta
import time

class MSRVTTDataset(Dataset):
    """
        videos_dir: directory where all videos are stored 
        config: AllConfig object
        split_type: 'train'/'test'
        img_transforms: Composition of transforms
    """
    def __init__(self, config, data, model, split_type = 'train', img_transforms=None, replayed_pairs=None):
        self.config = config
        self.videos_dir = config.videos_dir
        self.img_transforms = img_transforms
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model is not None:
            self.clip = model.to(self.device)
            self.clip.eval()
            
        self.data = data
        self.replayed_data = None
        self.auxiliary_pairs = None
        self.split_type = split_type
        db_file = 'datasets/MSRVTT/MSRVTT_data.json' 
        test_csv = 'datasets/MSRVTT/MSRVTT_CL_validation.csv'

        self.db = load_json(db_file)
        if self.split_type == 'train':
            if self.config.training_type == 'full_shot':
                self.videos = [video_id for category in self.data.values() for video_id in category]    
            elif self.config.training_type == 'few_shot':
                self.videos = []
                for category in self.data.values():
                    # Randomly select num_shots videos from each category
                    selected_videos = random.sample(category, min(self.config.num_shots, len(category)))
                    self.videos.extend(selected_videos)

            self._compute_vid2caption()
            self._construct_all_train_pairs()

            if replayed_pairs is not None:
                self.set_replayed_data(replayed_pairs)

        else:
            self.videos = [video_id for category in self.data.values() for video_id in category]
            self.test_df = pd.read_csv(test_csv)
    
    def __getitem__(self, index):
        video_path, text, video_id = self._get_vidpath_and_caption_by_index(index)

        frames, idxs = VideoCapture.load_frames(video_path, self.config.num_frames)

        # process images of video
        if self.img_transforms is not None:
            frames = self.img_transforms(frames)

        if self.split_type == 'train':
            if self.replayed_data is not None:
                # Randomly select an index for auxiliary data
                aux_index = self.get_next_auxiliary_index()
                image, caption = self.get_auxiliary_pairs_by_index(aux_index)
                return {
                    'video_id': video_id,
                    'video': frames,
                    'text': text,
                    'image': image,
                    'caption': caption,
                }
            else:
                return {
                    'video_id': video_id,
                    'video': frames,
                    'text': text,
                }
        else:
            return {
                'video_id': video_id,
                'video': frames,
                'text': text,
            }

    def __len__(self):
        if self.split_type == 'train':
            return len(self.all_train_pairs)

        return len(self.videos)

    def _get_vidpath_and_caption_by_index(self, index):
        # returns video path and caption as string
        if self.split_type == 'train':
            vid, caption = self.all_train_pairs[index]
            video_path = os.path.join(self.videos_dir, vid)
        else:
            vid = self.videos[index]
            video_path = os.path.join(self.videos_dir, vid)
            caption = self._get_sentence_for_video(vid, self.test_df)

        return video_path, caption, vid

    def _construct_all_train_pairs(self):
        self.all_train_pairs = []
        if self.split_type == 'train':
            for vid in self.videos:
                if self.config.benchmark == "para":
                    self.all_train_pairs.append([vid, self.vid2caption[vid]])
                else:
                    for caption in self.vid2caption[vid]:
                        self.all_train_pairs.append([vid, caption])
        
    def _compute_vid2caption(self):
        self.vid2caption = defaultdict(list)
        for annotation in self.db['sentences']:
            caption = annotation['caption']
            vid = annotation['video_id']
            if self.config.benchmark == "para":
                if self.vid2caption[vid]:
                    self.vid2caption[vid] += " " + caption
                else:
                    self.vid2caption[vid] = caption
            else:
                self.vid2caption[vid].append(caption)

    def _get_sentence_for_video(self, video_id, df):
        matching_row = df[df['video_id'] == video_id]
        if not matching_row.empty:
            return matching_row.iloc[0]['sentence']
        return None

    def _construct_auxiliary_pairs(self):
        start_time = time.time()
        print("Constructing auxiliary pairs...")
        image_transforms = init_transform_dict(self.config.input_res)['clip_test']
        self.auxiliary_pairs = []

        total_videos = len(self.videos)
        total_captions = sum(len(self.vid2caption[vid]) for vid in self.videos)
        processed_captions = 0
        for video_index, vid in enumerate(self.videos, 1):
            vid_path = os.path.join(self.videos_dir, vid)
            frames, idxs = VideoCapture.load_frames(vid_path, self.config.num_frames)
            frames = image_transforms(frames).to(self.device)
            # Reshape frames to [T, 1, C, H, W]
            frame_batch = frames.unsqueeze(1)
            vid_embed = self.clip.forward_video(frame_batch)

            for caption in self.vid2caption[vid]:
                text_token = clip_tokenizer(caption, return_tensors='pt', padding=True,
                                            truncation=True).to(self.device)
                text_embed = self.clip.forward_text(text_token.input_ids, text_token.attention_mask)

                similarity = torch.nn.functional.cosine_similarity(vid_embed, text_embed, dim=-1)
                max_similarity_index = torch.argmax(similarity)
                selected_frame = frames[max_similarity_index].to('cpu').detach()
                self.auxiliary_pairs.append([selected_frame, caption])

                processed_captions += 1

                # Update progress
                time_elapsed = time.time() - start_time
                print(f"\rProcessing: Video {video_index}/{total_videos} | "
                    f"Caption {processed_captions}/{total_captions} | "
                    f"Time: {timedelta(seconds=int(time_elapsed))} | "
                    f"Pairs: {len(self.auxiliary_pairs)}", end="")
                
        print("\nAuxiliary pairs constructed!")

    def get_next_auxiliary_index(self):
        if not self.available_aux_indices:
            # If all indices have been used, reset
            self.available_aux_indices = set(range(self.auxiliary_pairs_length))
            self.used_aux_indices.clear()

        # Randomly select an index from available indices
        aux_index = random.choice(list(self.available_aux_indices))
        self.available_aux_indices.remove(aux_index)
        self.used_aux_indices.add(aux_index)

        return aux_index

    def get_auxiliary_pairs_by_index(self, index):
        return self.replayed_data[index]

    def set_replayed_data(self, replayed_data):
        self.replayed_data = replayed_data

        self.auxiliary_pairs_length = len(self.replayed_data)
        self.available_aux_indices = set(range(self.auxiliary_pairs_length))
        self.used_aux_indices = set()