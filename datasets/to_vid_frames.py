import os
import cv2
import random
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

def preprocess_video(video_path, output_dir, num_frames, sample='uniform'):
    video_name = Path(video_path).stem
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), f"Cannot open video file: {video_path}"
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    acc_samples = min(num_frames, vlen)
    intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
    ranges = []
    
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
    if sample == 'rand':
        frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
    else:  # sample == 'uniform':
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]

    frames = []
    for i, index in enumerate(frame_idxs, 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
        if not ret:
            n_tries = 5
            for _ in range(n_tries):
                ret, frame = cap.read()
                if ret:
                    break
        if ret:
            img_path = os.path.join(video_output_dir, f'frame_{i}_idx_{index:04d}.png')
            cv2.imwrite(img_path, frame)
            frames.append(frame)
        else:
            raise ValueError(f"Could not read frame {index} from video {video_path}")
        
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())
        img_path = os.path.join(video_output_dir, f'frame_{len(frames)}_idx_{index:04d}.png')
        cv2.imwrite(img_path, frames[-1])
                
    cap.release()
    return f"Processed: {video_path}"

def process_video_wrapper(args):
    try:
        return preprocess_video(*args)
    except Exception as e:
        return f"Error processing {args[0]}: {str(e)}"

def main():
    video_dir = "/home/uqzzha35/Project/CL_Video_Retrieval/CL_for_VTR/datasets/msrvtt_data/MSRVTT_Videos"
    output_dir = "/home/uqzzha35/Project/CL_Video_Retrieval/CL_for_VTR/datasets/msrvtt_data/MSRVTT_Frames6"
    num_frames = 6
    sample_method = "uniform"
    
    video_paths = sorted(Path(video_dir).glob("*.mp4"))
    
    if not video_paths:
        print(f"No .mp4 files found in {video_dir}")
        return

    print(f"Found {len(video_paths)} .mp4 files in {video_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Frames per video: {num_frames}")
    print(f"Sampling method: {sample_method}")
    print()

    args_list = [(str(video_path), output_dir, num_frames, sample_method) for video_path in video_paths]

    num_processes = multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(executor.map(process_video_wrapper, args_list))

    for result in results:
        print(result)

    print("All videos processed!")

if __name__ == "__main__":
    main()