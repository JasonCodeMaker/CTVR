from datasets.model_transforms import init_transform_dict
from datasets.data_factory import DataFactory
import random

from modules.tokenizer import clip_tokenizer
from datasets.video_capture import VideoCapture
from datetime import timedelta
import time
import os
import torch


class CILSetTask:
    def __init__(self, set_tasks, model, split_type, dataset_name, config):
        self.num_tasks = len(set_tasks) + 1

        self.split_type = split_type
        if split_type == 'train':
            self.shuffle = True
            if config.training_type == 'full_shot':
                self.set_tasks = set_tasks
            elif config.training_type == 'few_shot':
                self.set_tasks = self.few_shot_selection(set_tasks, config)
            else:
                raise NotImplementedError

        else:
            self.shuffle = False
            self.set_tasks = set_tasks

        self.config = config
        self.model = model

        self.train_enable = True
        self.dataset_name = dataset_name
        self.current_task = 1
        self.current_task_dataset = None

        self.replayed_pairs = None
    
        img_transforms = init_transform_dict(self.config.input_res)
        if split_type == 'train':
            self.img_transforms = img_transforms['clip_train']
        else:
            self.img_transforms = img_transforms['clip_test']

    def __iter__(self):
        self.memory = {}
        self.current_task_dataset = None
        self.current_task = 1
        return self

    def get_dataloader(self, data, memory = None):
        if memory != None:
            data = {**memory, **data}

        dataloader = DataFactory.get_data_loader(self.config, self.dataset_name, data, self.model, self.split_type, self.img_transforms)

        return dataloader
    
    def set_memory(self, memory):
        self.memory = memory
        
    def set_model(self, model):
        self.model = model        

    def __next__(self):
        data = self.set_tasks[self.current_task]

        if self.train_enable:
            comp_data = {**self.memory, **data}
        else:
            comp_data = data
        
        self.current_task_dataloader, dataset = DataFactory.get_data_loader(self.config, self.dataset_name, 
                                                                            comp_data, self.model, self.split_type, self.img_transforms, self.replayed_pairs)

        self.current_task += 1
        return comp_data, self.current_task_dataloader, dataset
    
    def get_valSet_by_taskNum(self, num_task):
        eval_data = {}
        total_data = []
        list_val_loaders = []
        list_num_classes = []
        for k in range(1, num_task):
            data = self.set_tasks[k]
            eval_data = {**eval_data, **data}
            total_data.append(data)
            list_num_classes.append(len(data.keys()))

        for i, data_i in enumerate(total_data):
            val_task_dataloader = DataFactory.get_data_loader(self.config, self.dataset_name, data_i, self.model, self.split_type, self.img_transforms)
            list_val_loaders.append((val_task_dataloader, list_num_classes[i]))
        return list_val_loaders
    
    def few_shot_selection(self, set_tasks, config):
        few_shot_tasks = {}
        
        for task, categories in set_tasks.items():
            few_shot_tasks[task] = {}
            
            for category, samples in categories.items():
                num_samples = min(config.num_shots, len(samples))       
                # Randomly select num_samples from the category
                selected_samples = random.sample(samples, num_samples)              
                # Add the selected samples to the few_shot_tasks dictionary
                few_shot_tasks[task][category] = selected_samples
        
        return few_shot_tasks

    def construct_replayed_data(self, model, dataset, device):
        # Step 1: Construct video-caption pairs
        videos_dir = dataset.videos_dir
        videos = dataset.videos
        vid2caption = dataset.vid2caption

        # Step 2: Construct best frame-caption pairs
        start_time = time.time()
        print("Constructing best frame-caption pairs...")
        image_transforms = init_transform_dict(self.config.input_res)['clip_test']
        replayed_pairs = []

        total_videos = len(videos)
        for video_index, vid in enumerate(videos, 1):
            vid_path = os.path.join(videos_dir, vid)
            frames, idxs = VideoCapture.load_frames(vid_path, self.config.num_frames)
            frames = image_transforms(frames).to(device)
            # Reshape frames to [T, 1, C, H, W]
            frame_batch = frames.unsqueeze(1)
            vid_embed = model.forward_video(frame_batch)

            # Process all captions for this video in a single batch
            captions = vid2caption[vid]
            text_tokens = clip_tokenizer(captions, return_tensors='pt', padding=True, truncation=True).to(device)
            text_embeds = model.forward_text(text_tokens.input_ids, text_tokens.attention_mask)

            # Calculate similarities between all frames and all captions
            similarities = torch.nn.functional.cosine_similarity(vid_embed.unsqueeze(1), text_embeds.unsqueeze(0), dim=-1)
            
            # Find the best frame-caption pair
            max_similarity = torch.max(similarities)
            max_indices = torch.where(similarities == max_similarity)
            best_frame_idx, best_caption_idx = max_indices[0][0], max_indices[1][0]

            best_frame = frames[best_frame_idx].to('cpu').detach()
            best_caption = captions[best_caption_idx]

            replayed_pairs.append([best_frame, best_caption])

            # Update progress
            time_elapsed = time.time() - start_time
            print(f"\rProcessing: Video {video_index}/{total_videos} | "
                f"Time: {timedelta(seconds=int(time_elapsed))} | "
                f"Pairs: {len(replayed_pairs)}", end="")

        print("\n Replayed pairs constructed!")
        if self.replayed_pairs == None:
            self.replayed_pairs = replayed_pairs
        else:
            self.replayed_pairs += replayed_pairs