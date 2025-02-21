import time
from datetime import timedelta
import torch
import os
import numpy as np
import csv

def log_training_progress(experiment, task_id, step, epoch, batch_idx, num_steps, epoch_start_time, full_loss, current_lr):
    experiment.log_metric(f'lr_task_{task_id}', current_lr, step=step)
    experiment.log_metric(f'loss_task_{task_id}', full_loss, step=step)
    
    time_elapsed = time.time() - epoch_start_time
    print(f"\rTrain Epoch: {epoch} | Batch: {batch_idx+1}/{num_steps} | "
          f"Time: {timedelta(seconds=int(time_elapsed))} | "
          f"Full Loss: {full_loss:.3f} | "
          f"LR: {current_lr:.2e}", end="")

def log_training_progress_img(experiment, task_id, step, epoch, batch_idx, num_steps, epoch_start_time, full_loss, img_loss, current_lr, lr=None):
    if lr is not None:
        experiment.log_metric(f'lr_task_{task_id}_{lr}', current_lr, step=step)
        experiment.log_metric(f'loss_task_{task_id}_{lr}', full_loss, step=step)
        experiment.log_metric(f'moe_loss_task_{task_id}_{lr}', img_loss, step=step)
    else:
        experiment.log_metric(f'lr_task_{task_id}', current_lr, step=step)
        experiment.log_metric(f'loss_task_{task_id}', full_loss, step=step)
        experiment.log_metric(f'moe_loss_task_{task_id}', img_loss, step=step)
    
    time_elapsed = time.time() - epoch_start_time
    print(f"\rTrain Epoch: {epoch} | Batch: {batch_idx+1}/{num_steps} | "
          f"Time: {timedelta(seconds=int(time_elapsed))} | "
          f"Full Loss: {full_loss:.3f} | "
          f"MoE Loss: {img_loss:.3f} | "
          f"LR: {current_lr:.2e}", end="")

def log_validation_progress(epoch, batch_idx, num_batches, validation_start_time):
    time_elapsed = time.time() - validation_start_time
    print(f"\rValidate Epoch: {epoch} | Batch: {batch_idx+1}/{num_batches} | "
          f"Time: {timedelta(seconds=int(time_elapsed))}", end="")

def log_validation_results(experiment, task_id, step, epoch, res, config):
    print(f"\n----------Testing Results of Task {task_id} at epoch {epoch} step {step}----------\n",
          f"R@1: {res['R1']} \n", 
          f"R@5: {res['R5']} \n", 
          f"R@10: {res['R10']} \n",
          f"MedR: {res['MedR']} \n",
          f"MeanR: {res['MeanR']} \n")
    print("--------------------------------------------------\n")

    # if curr_step == total_steps:
    #     experiment.log_metric(f"R@1", res['R1'], step=(task_id-1) * config.max_num_epochs + epoch)
    # if task_id == 1:
    #     experiment.log_metric(f"R@1_of_Task{task_id}", res['R1'], step=step)

def log_final_validation_progress(n_task, total_tasks, batch_idx, num_batches, 
                                  task_start_time, validation_start_time):
    task_time_elapsed = time.time() - task_start_time
    total_time_elapsed = time.time() - validation_start_time
    print(f"\rFinal Validate Task: {n_task+1}/{total_tasks} | Batch: {batch_idx+1}/{num_batches} | "
          f"Task Time: {timedelta(seconds=int(task_time_elapsed))} | "
          f"Total Time: {timedelta(seconds=int(total_time_elapsed))}" , end="")

def save_video_embeddings(checkpoint_dir, task_id, vid_embeds_pooled, lr=None):
    save_dir = os.path.join(checkpoint_dir, 'database')
    os.makedirs(save_dir, exist_ok=True)

    if lr is not None:
        filename = os.path.join(save_dir, f'task{task_id}_vid_embed_lr_{lr}.pth')
    else:
        filename = os.path.join(save_dir, f'task{task_id}_vid_embed.pth')
    torch.save(vid_embeds_pooled, filename)
    print(f"\nSaving video pooled embeddings: {filename} ...")

def save_task_prototype(checkpoint_dir, task_id, prototype):
    save_dir = os.path.join(checkpoint_dir, 'prototypes')
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f'task{task_id}_prototype.pth')
    torch.save(prototype, filename)
    print(f"Saving task prototype: {filename} ...")

def store_route_weight(route_weights, checkpoint_dir, task_id):
    save_dir = os.path.join(checkpoint_dir, 'route_weights')
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f'task{task_id}_route_weights.pth')
    torch.save(route_weights, filename)
    print(f"Saving route weights for task {task_id}: {filename} ...")

def load_stored_embed(store_dir, task_id):
    filename = os.path.join(store_dir, f'database/task{task_id}_vid_embed.pth')
    print(f"\nLoading stored video pooled embeddings from {filename} ...")
    return torch.load(filename, weights_only=True)

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def visulize_path(array):
    if array.size == 0:
        print("Empty array")
        return
    
    # Ensure the array is 2D
    if array.ndim == 1:
        array = array.reshape(1, -1)
    
    # Find the maximum width needed for any element
    max_width = len(str(np.max(array)))
    
    print("\nTraining path:")
    # Create the top border
    border = '+' + ('-' * (max_width + 2) + '+') * array.shape[1]
    
    print(border)
    for row in array:
        # Print each cell with vertical borders
        print('|', end='')
        for element in row:
            print(f' {element:>{max_width}} |', end='')
        print()  # New line after each row
        print(border)  # Bottom border for each row

def track_expert_status(save_dir, frozen_path, num_layers, num_experts, task_id):
    # Initialize path lists for vision and text models
    vision_path_list = np.zeros((num_layers, num_experts), dtype=int)
    text_path_list = np.zeros((num_layers, num_experts), dtype=int)
    
    # Read frozen layers from file
    with open(frozen_path, "r") as file:
        frozen_layers = file.read().splitlines()
    
    # Process each frozen layer
    for layer in frozen_layers:
        parts = layer.split('.')
        if parts[1] == 'vision_model':
            model_type = 'vision'
            layer_idx = int(parts[4])
            expert_idx = int(parts[6])
        else:
            model_type = 'text_model'
            layer_idx = int(parts[4])
            expert_idx = int(parts[6])
        
        # Mark the expert as frozen (0)
        if model_type == 'vision':
            vision_path_list[layer_idx, expert_idx] = 1
        else:
            text_path_list[layer_idx, expert_idx] = 1
            
    path_dir = os.path.join(save_dir, 'path')
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    
    vision_path_file = os.path.join(path_dir, f'vision_path_{task_id}.npy')
    text_path_file = os.path.join(path_dir, f'text_path_{task_id}.npy')
    
    np.save(vision_path_file, vision_path_list)
    np.save(text_path_file, text_path_list)

    return vision_path_list, text_path_list

def construct_exp_log(filename, num_tasks=20, lr=None):
    headers = ["Task", "R@1", "R@5", "R@10", "MedR", "MeanR", "BWF", "Lr"]
    if lr is not None:
        headers.append("Lr")

    data = [[i+1] + ["" for _ in range(len(headers)-1)] for i in range(num_tasks)]
    
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data)
    
    print(f"Empty CSV file '{filename}' has been created successfully.")

def update_exp_result(filename, task, r1=None, r5=None, r10=None, medr=None, meanr=None, bwf=None, lr=None):
    # Read the existing CSV file
    with open(filename, 'r', newline='') as file:
        reader = csv.reader(file)
        data = list(reader)
    
    # Update the specified row
    if 1 <= task <= len(data) - 1:
        row = data[task]
        if r1 is not None: row[1] = f"{round(float(r1), 2):.2f}"
        if r5 is not None: row[2] = f"{round(float(r5), 2):.2f}"
        if r10 is not None: row[3] = f"{round(float(r10), 2):.2f}"
        if medr is not None: row[4] = f"{round(float(medr), 2):.2f}"
        if meanr is not None: row[5] = f"{round(float(meanr), 2):.2f}"
        if bwf is not None: row[6] = f"{round(float(bwf), 2):.2f}"
        if lr is not None: row[7] = f"{lr:.2e}"
    
    # Write the updated data back to the CSV file
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    
    print(f"Task {task} has been updated in '{filename}'.")

def construct_task_log(filename, lr_list):
    headers = ["Lr", "R@1", "R@5", "R@10", "MedR", "MeanR", "BWF"]

    # Prepare data with each learning rate in a new row
    data = [[lr] + [""] * (len(headers) - 1) for lr in lr_list]

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data)

    print(f"CSV file '{filename}' with learning rates has been created successfully.")

import csv

def update_task_result(filename, r1=None, r5=None, r10=None, medr=None, meanr=None, bwf=None, lr=None):
    # Read the existing CSV file
    with open(filename, 'r', newline='') as file:
        reader = csv.reader(file)
        data = list(reader)
    
    # Find the index of the "Lr" column (assumed to be the first row)
    lr_index = data[0].index("Lr")
    
    # Find the row corresponding to the specified `lr` value
    for row in data[1:]:  # Skip header
        if row[lr_index] == str(lr):  # Compare as string for consistency
            # Update the corresponding columns if values are provided
            if r1 is not None: row[1] = f"{float(r1):.2f}"
            if r5 is not None: row[2] = f"{float(r5):.2f}"
            if r10 is not None: row[3] = f"{float(r10):.2f}"
            if medr is not None: row[4] = f"{float(medr):.2f}"
            if meanr is not None: row[5] = f"{float(meanr):.2f}"
            if bwf is not None: row[6] = f"{float(bwf):.2f}"
        
    # Write the updated data back to the CSV file
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

    print(f"CSV file '{filename}' has been updated successfully for Lr = {lr}.")
