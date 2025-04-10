import os
from modules.basic_utils import load_json
from torch.utils.data import Dataset
from datasets.video_capture import VideoCapture


class ACTNETDataset(Dataset):
    """
        videos_dir: directory where all videos are stored 
        config: AllConfig object
        split_type: 'train'/'test'
        img_transforms: Composition of transforms
    """
    def __init__(self, config, data, model, split_type = 'train', img_transforms=None):
        self.config = config
        self.videos_train_dir = os.join(config.videos_dir, 'train')
        self.videos_val_dir = os.join(config.videos_dir, 'val')
        self.img_transforms = img_transforms
        self.data = data
        self.split_type = split_type

        # get all video ids for training or testing
        self.videos = [video_id for category in self.data.values() for video_id in category]

        # Create video_id to category mapping
        self.video_to_category = {}
        for category, videos in self.data.items():
            for video_id in videos:
                if self.config.benchmark == 'anet_clip':
                    video_id = video_id + '_1'
                self.video_to_category[video_id] = category

        if self.config.benchmark == 'anet_cap':
            db_file = 'datasets/ACTNET/queries.json'
            self.vid2caption = load_json(db_file)
            if split_type == 'train':
                self._construct_all_train_pairs('anet_cap')
            else:
                self._construct_all_test_pairs('anet_cap')
        elif self.config.benchmark == 'anet_para':
            db_file = 'datasets/ACTNET/queries.json'
            self.vid2caption = load_json(db_file)
            if split_type == 'train':
                self._construct_all_train_pairs('anet_para')
            else:
                self._construct_all_test_pairs('anet_para')  
        elif self.config.benchmark == 'anet_clip':
            if self.split_type == 'train':
                db_file = 'datasets/ACTNET/filtered_train_queries_5.json'
                self.vid2caption = load_json(db_file)
                self._construct_all_train_pairs('anet_clip')
            else:
                db_file = 'datasets/ACTNET/filtered_val_queries.json'
                self.vid2caption = load_json(db_file)
                self._construct_all_test_pairs('anet_clip')
        else:
            raise ValueError
    
    def __getitem__(self, index):
        if self.split_type == 'train':
            video_path, caption, video_id = self._get_vidpath_and_caption_by_index_train(index)
        else:
            video_path, caption, video_id = self._get_vidpath_and_caption_by_index_test(index)

        imgs, idxs = VideoCapture.load_frames(video_path, self.config.num_frames)

        # process images of video
        if self.img_transforms is not None:
            imgs = self.img_transforms(imgs)

        # Get the category of the video
        category = self.video_to_category[video_id]

        return {
            'video_id': video_id,
            'video': imgs,
            'text': caption,
            'category': category, 
        }


    def _get_vidpath_and_caption_by_index_train(self, index):
        vid, caption = self.all_train_pairs[index]
        video_path = os.path.join(self.videos_train_dir, vid)
        return video_path, caption, vid

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