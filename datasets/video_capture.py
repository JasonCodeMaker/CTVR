import cv2
import random
import numpy as np
import torch
from pathlib import Path

class VideoCapture:

    @staticmethod
    def load_frames_from_video(video_path,
                               num_frames,
                               sample='rand'):
        """
            video_path: str/os.path
            num_frames: int - number of frames to sample
            sample: 'rand' | 'uniform' how to sample
            returns: frames: torch.tensor of stacked sampled video frames 
                             of dim (num_frames, C, H, W)
                     idxs: list(int) indices of where the frames where sampled
        """
        cap = cv2.VideoCapture(video_path)
        assert (cap.isOpened()), video_path
        vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # get indexes of sampled frames
        acc_samples = min(num_frames, vlen)
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []

        # ranges constructs equal spaced intervals (start, end)
        # we can either choose a random image in the interval with 'rand'
        # or choose the middle frame with 'uniform'
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
        else:  # sample == 'uniform':
            frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]

        frames = []
        for index in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, frame = cap.read()
            if not ret:
                n_tries = 5
                for _ in range(n_tries):
                    ret, frame = cap.read()
                    if ret:
                        break
            if ret:
                #cv2.imwrite(f'images/{index}.jpg', frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame)
                # (H x W x C) to (C x H x W)
                frame = frame.permute(2, 0, 1)
                frames.append(frame)
            else:
                raise ValueError

        while len(frames) < num_frames:
            frames.append(frames[-1].clone())
            
        frames = torch.stack(frames).float() / 255
        cap.release()
        return frames, frame_idxs

    @staticmethod
    def load_frames(directory_path, num_frames=None):
        directory = Path(directory_path)
        frame_files = list(directory.glob('frame_*.png'))
        
        # Sort frame_files based on frame_idx
        frame_files.sort(key=lambda x: int(x.stem.split('_')[1]))
        
        if num_frames is not None:
            frame_files = frame_files[:num_frames]
        
        frames = []
        frame_idxs = []
        
        for frame_file in frame_files:
            frame_idx = int(frame_file.stem.split('_')[1])
            frame_idxs.append(frame_idx)
            
            frame = cv2.imread(str(frame_file))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            
            frame = torch.from_numpy(frame)
            frame = frame.permute(2, 0, 1)
            frames.append(frame)
        
        frames = torch.stack(frames).float() / 255
        return frames, frame_idxs

    @staticmethod
    def load_frames_from_fps(directory_path, num_frames=None, sample='unifrom'):
        directory = Path(directory_path)
        frame_files = list(directory.glob('frame_*.png'))
        
        # Sort frame_files based on frame_idx
        frame_files.sort(key=lambda x: int(x.stem.split('_')[1]))

        total_frames = len(frame_files)
        
        if num_frames is not None and num_frames < total_frames:
            if sample == 'uniform':
                intervals = np.linspace(start=0, stop=total_frames, num=num_frames + 1).astype(int)
                selected_indices = [(intervals[i] + intervals[i+1] - 1) // 2 for i in range(num_frames)]
                frame_files = [frame_files[i] for i in selected_indices]
            elif sample == 'rand':
                intervals = np.linspace(start=0, stop=total_frames, num=num_frames + 1).astype(int)
                selected_indices = [np.random.randint(intervals[i], intervals[i+1]) for i in range(num_frames)]
                frame_files = [frame_files[i] for i in selected_indices]
            else:
                frame_files = frame_files[:num_frames] 
        elif num_frames > total_frames:
            # Repeat frames from the beginning until we reach num_frames
            frame_files = frame_files * (num_frames // total_frames) + frame_files[:num_frames % total_frames]
        
        frames = []
        frame_idxs = []
        
        for frame_file in frame_files:
            frame_idx = int(frame_file.stem.split('_')[1])
            frame_idxs.append(frame_idx)
            
            frame = cv2.imread(str(frame_file))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            
            frame = torch.from_numpy(frame)
            frame = frame.permute(2, 0, 1)
            frames.append(frame)
        
        frames = torch.stack(frames).float() / 255
        return frames, frame_idxs