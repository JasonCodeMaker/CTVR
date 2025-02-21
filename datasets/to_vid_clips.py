import os
import cv2
import json
import random
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np


def extract_frames_from_clip(cap, start_frame, end_frame, num_frames, sample='uniform'):
    clip_length = end_frame - start_frame
    acc_samples = min(num_frames, clip_length)
    intervals = np.linspace(start=start_frame, stop=end_frame, num=acc_samples + 1).astype(int)
    ranges = []
    
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
    if sample == 'random':
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
            frames.append(frame)
        else:
            raise ValueError(f"Could not read frame {index}")
        
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())
                
    return frames

def clip_already_processed(clip_output_dir, num_frames_per_clip):
    if not os.path.exists(clip_output_dir):
        return False
    
    # Check if the expected number of frames exists
    expected_frames = [f'frame_{i:03d}.png' for i in range(1, num_frames_per_clip + 1)]
    return all(os.path.exists(os.path.join(clip_output_dir, frame)) for frame in expected_frames)


def preprocess_video(video_path, output_dir, timestamps, num_frames_per_clip=24, sample='uniform'):
    video_name = Path(video_path).stem
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return f"Error: Cannot open video file: {video_path}"
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for clip_idx, (start_time, end_time) in enumerate(timestamps, 1):
        clip_output_dir = os.path.join(output_dir, f"{video_name}_{clip_idx}")

        if clip_already_processed(clip_output_dir, num_frames_per_clip):
            continue  # Skip this clip if it's already processed

        os.makedirs(clip_output_dir, exist_ok=True)

        start_frame = min(int(start_time * fps), max_frames - 1)
        end_frame = min(int(end_time * fps), max_frames)
        
        if start_frame >= end_frame:
            print(f"Warning: Invalid time range for clip {clip_idx} of {video_path}. Skipping this clip.")
            continue
        
        try:
            clip_frames = extract_frames_from_clip(cap, start_frame, end_frame, num_frames_per_clip, sample)
            
            for frame_idx, frame in enumerate(clip_frames, 1):
                img_path = os.path.join(clip_output_dir, f'frame_{frame_idx:03d}.png')
                cv2.imwrite(img_path, frame)
        except ValueError as e:
            print(f"Error processing clip {clip_idx} of {video_path}: {str(e)}")
            print(f"Clip time range: {start_time:.2f} - {end_time:.2f}")
            print(f"Attempted frame range: {start_frame} - {end_frame}")
            print(f"Max frames in video: {max_frames}")
    
    cap.release()
    return f"Processed: {video_path}"

def process_video_wrapper(args):
    try:
        return preprocess_video(*args)
    except Exception as e:
        return f"Error processing {args[0]}: {str(e)}"

def main():
    video_dir = "/home/uqzzha35/Project/CL_Video_Retrieval/CL_for_VTR/datasets/ActivityNet/Activity_Videos"
    output_dir = "/home/uqzzha35/Project/CL_Video_Retrieval/CL_for_VTR/datasets/ActivityNet/Activity_Clip_Frames"
    json_file = "/home/uqzzha35/Project/CL_Video_Retrieval/CL_for_VTR/datasets/ActivityNet/train.json"
    num_frames_per_clip = 24
    sample_method = "uniform"  # or "random"
    
    # Load JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    video_paths = []
    args_list = []
    
    for video_id, video_info in data.items():
        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        if os.path.exists(video_path):
            video_paths.append(video_path)
            timestamps = video_info['timestamps']
            args_list.append((video_path, output_dir, timestamps, num_frames_per_clip, sample_method))
    
    if not video_paths:
        print(f"No matching video files found in {video_dir}")
        return

    print(f"Found {len(video_paths)} matching video files in {video_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Frames per clip: {num_frames_per_clip}")
    print(f"Sampling method: {sample_method}")
    print()

    num_processes = multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(executor.map(process_video_wrapper, args_list))

    # for result in results:
    #     if result is not None:
    #         print(result)

    print("All videos processed!")

if __name__ == "__main__":
    main()