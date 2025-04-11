import torch
import time
import os
from modules.metrics import text_embed_processing
from modules.trainer_utils import (
    log_validation_progress, log_validation_results,
    log_final_validation_progress, save_video_embeddings, load_stored_embed,
    save_task_prototype, AverageMeter, update_exp_result
)

class Validator:
    def __init__(self, model, metrics, config, task_id, val_loaders_list, tokenizer, accelerator, experiment, checkpoint_dir, list_val_acc_ii, task_log, overall_log):
        self.model = model
        self.metrics = metrics
        self.config = config
        self.task_id = task_id
        self.val_loaders_list = val_loaders_list
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.experiment = experiment
        self.checkpoint_dir = checkpoint_dir
        self.list_val_acc_ii = list_val_acc_ii
        self.task_log = task_log
        self.overall_log = overall_log
        self.device = accelerator.device

    def _feature_extraction(self, data, prototype_id=None):
        if self.tokenizer is not None:
            data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
        if isinstance(data['text'], torch.Tensor):
            data['text'] = data['text'].to(self.device)
        else:
            data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}
        data['video'] = data['video'].to(self.device)
        if prototype_id is not None:
            data['prototype_id'] = prototype_id
        text_embed, vid_embed = self.model(data)
        return text_embed, vid_embed

    def _proto_feature_extraction(self, data, return_vid=False):
        if self.tokenizer is not None:
            data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
        if isinstance(data['text'], torch.Tensor):
            data['text'] = data['text'].to(self.device)
        else:
            data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}
        data['video'] = data['video'].to(self.device)
        text_embed = self.model.forward_text(data['text']['input_ids'], data['text']['attention_mask'], self.task_id)
        if return_vid:    
            vid_embed = self.model.forward_video(data['video'])
            return text_embed, vid_embed
        return text_embed          

    def task_validation(self, task_id, epoch):
        """Validate on the current task."""
        self.model.eval()
        text_embed_arr = []
        vid_embed_arr = []
        all_vid_ids = []
        validation_start_time = time.time()
        if epoch == 0:
            print("Initial Validation Start...")
        else:
            print("\nValidate on the current task...")
        with self.experiment.validate():
            with torch.no_grad():
                val_loader, num_classes = self.val_loaders_list[task_id - 1]
                prototype_id = task_id if self.config.task_prototype else None
                num_batches = len(val_loader)
                # Extract all video and text embeddings for the current task
                for batch_idx, data in enumerate(val_loader):
                    text_embed, vid_embed = self._feature_extraction(data, prototype_id=prototype_id)
                    text_embed_arr.append(text_embed)
                    vid_embed_arr.append(vid_embed)
                    for v_id in data['video_id']:
                        all_vid_ids.append(v_id)
                    log_validation_progress(epoch, batch_idx, num_batches, validation_start_time)
                text_embeds = torch.cat(text_embed_arr) # (num_samples, embed_dim)
                vid_embeds = torch.cat(vid_embed_arr) # (num_samples, embed_dim)
                
                # Remove duplicate videos if multiple captions per video (If one-to-many mapping)
                vid_embeds_per_video_id = {}
                for idx, v_id in enumerate(all_vid_ids):
                    if v_id not in vid_embeds_per_video_id:
                        vid_embeds_per_video_id[v_id] = vid_embeds[idx]
                vid_embeds = torch.stack([vid_embeds_per_video_id[v_id] for v_id in vid_embeds_per_video_id])

                # Process text embeddings for each video
                text_embeds_per_video_id = text_embed_processing(text_embeds, all_vid_ids, 1)

                # Calculate similarity scores and measure performance
                sims = text_embeds_per_video_id @ vid_embeds.t()
                res = self.metrics(sims)
                print(f"\n-----Task Val Epoch: {epoch}-----\n"
                      f"R@1: {res['R1']}\n"
                      f"R@5: {res['R5']}\n"
                      f"R@10: {res['R10']}\n"
                      f"MedR: {res['MedR']}\n"
                      f"MeanR: {res['MeanR']}")
                self.experiment.log_metric(f"R@1_of_Task{task_id}", res['R1'])

        return res

    def validate(self, task_id, epoch, step):
        """General validation that considers all tasks."""
        self.model.eval()
        text_embed_arr = []
        vid_embed_arr = []
        all_vid_ids = []
        curr_vid_ids = []
        num_batches = 0
        batch_indices = 0
        validation_start_time = time.time()
        print("\nValidate on all data...")
        with self.experiment.validate():
            with torch.no_grad():
                for n_task, (val_loader, num_classes) in enumerate(self.val_loaders_list):
                    num_batches += len(val_loader)
                    for data in val_loader:
                        if n_task == task_id - 1:
                            if self.config.task_prototype:
                                text_embed, vid_embed = self._proto_feature_extraction(data, return_vid=True)
                            else:
                                text_embed, vid_embed = self._feature_extraction(data)
                        else:
                            if self.config.task_prototype:
                                text_embed = self._proto_feature_extraction(data)
                            else:
                                text_embed, _ = self._feature_extraction(data)
                        text_embed_arr.append(text_embed)
                        if n_task == task_id - 1:
                            vid_embed_arr.append(vid_embed)
                            curr_vid_ids.extend(data['video_id'])
                        for v_id in data['video_id']:
                            all_vid_ids.append(v_id)
                        log_validation_progress(epoch, batch_indices, num_batches, validation_start_time)
                        batch_indices += 1

                if self.config.task_prototype:
                    text_embeds = torch.cat(text_embed_arr, dim=1)
                else:
                    text_embeds = torch.cat(text_embed_arr)

                if len(vid_embed_arr) == 0:
                    raise RuntimeError(f"vid_embed_arr is empty before torch.cat. This likely means no video embeddings were collected for the current task (n_task={n_task}, task_id={task_id}). Check the logic above for populating vid_embed_arr.")
                curr_vid_embeds = torch.cat(vid_embed_arr)
                # Remove duplicate video embeddings
                vid_embeds_per_video_id = {}
                for idx, v_id in enumerate(curr_vid_ids):
                    if v_id not in vid_embeds_per_video_id:
                        vid_embeds_per_video_id[v_id] = curr_vid_embeds[idx]
                curr_vid_embeds = torch.stack([vid_embeds_per_video_id[v_id] for v_id in vid_embeds_per_video_id])

                # Load stored video embeddings from previous tasks
                vid_embeds_arr = []
                if task_id > 1:
                    for n_task in range(task_id - 1):
                        task_vid_embeds = load_stored_embed(self.checkpoint_dir, n_task + 1)
                        vid_embeds_arr.append(task_vid_embeds.to(self.device))
                vid_embeds_arr.append(curr_vid_embeds)

                if self.config.task_prototype:
                    total_sims = []
                    for prototype_id in range(task_id):
                        task_text_embeds = text_embeds[prototype_id]
                        text_embeds_per_video_id = text_embed_processing(task_text_embeds, all_vid_ids, 1)
                        task_sims = text_embeds_per_video_id @ vid_embeds_arr[prototype_id].t()
                        total_sims.append(task_sims)
                    sims = torch.cat(total_sims, dim=-1)
                else:
                    total_vid_embeds = torch.cat(vid_embeds_arr)
                    text_embeds_per_video_id = text_embed_processing(text_embeds, all_vid_ids, 1)
                    sims = text_embeds_per_video_id @ total_vid_embeds.t()

                res = self.metrics(sims)
                log_validation_results(self.experiment, task_id, step, epoch, res, self.config)
        return res

    def final_validate(self, task_id, list_val_acc_ii):
        """Final evaluation at the end of training for all tasks."""
        if self.config.load_best:
            checkpoint_path = f'task{task_id}_model_best.pth'
            self._load_checkpoint(checkpoint_path)
        self.model.eval()
        BWF = AverageMeter()
        validation_start_time = time.time()
        total_tasks = len(self.val_loaders_list)

        print("\nFinal Validation Start...")
        with self.experiment.validate():
            with torch.no_grad():
                for n_task, (val_loader, num_classes) in enumerate(self.val_loaders_list):
                    task_start_time = time.time()
                    prototype_id = n_task + 1 if self.config.task_prototype else None
                    num_batches = len(val_loader)
                    text_embed_arr = []
                    vid_embed_arr = []
                    all_vid_ids = []
                    print(f"Task: {n_task + 1}/{total_tasks}")
                    for batch_idx, data in enumerate(val_loader):
                        text_embed, vid_embed = self._feature_extraction(data, prototype_id)
                        text_embed_arr.append(text_embed)
                        vid_embed_arr.append(vid_embed)
                        for v_id in data['video_id']:
                            all_vid_ids.append(v_id)
                        log_final_validation_progress(n_task, total_tasks, batch_idx, num_batches, task_start_time, validation_start_time)
                    text_embeds = torch.cat(text_embed_arr)
                    vid_embeds = torch.cat(vid_embed_arr)
                    # Remove duplicate video embeddings
                    vid_embeds_per_video_id = {}
                    for idx, v_id in enumerate(all_vid_ids):
                        if v_id not in vid_embeds_per_video_id:
                            vid_embeds_per_video_id[v_id] = vid_embeds[idx]
                    vid_embeds = torch.stack([vid_embeds_per_video_id[v_id] for v_id in vid_embeds_per_video_id])
                    if n_task < task_id - 1:
                        vid_embeds = load_stored_embed(self.checkpoint_dir, n_task + 1)
                    else:
                        save_video_embeddings(self.checkpoint_dir, n_task + 1, vid_embeds)
                        if self.config.task_prototype:
                            prototype = self.model.clipmodel.text_model.task_prototype[n_task]
                            save_task_prototype(self.checkpoint_dir, n_task + 1, prototype)
                    text_embeds_per_video_id = text_embed_processing(text_embeds, all_vid_ids, 1)
                    sims = text_embeds_per_video_id @ vid_embeds.t()
                    res = self.metrics(sims)
                    if n_task == task_id - 1:
                        list_val_acc_ii.append(res['R1'])
                        print(f"\nValidation R@1 List: {list_val_acc_ii}")
                        print(f"Task {n_task + 1} R@1: {res['R1']:.6f}")
                    elif n_task < task_id - 1:
                        if len(list_val_acc_ii) == 0:
                            print(f"Task {n_task + 1} R@1: {res['R1']:.6f}")
                        else:
                            forgetting = list_val_acc_ii[n_task] - res['R1']
                            print(f"\nValidation R@1 List: {list_val_acc_ii}")
                            print(f"Task {n_task + 1} R@1: {res['R1']:.6f}")
                            print(f"Task {n_task + 1} Forgetting: {forgetting:.6f}")
                            BWF.update(forgetting, num_classes)
                            print(f"Task {n_task + 1} BWF: {BWF.avg:.6f}")
        update_exp_result(self.task_log, task_id, bwf=BWF.avg)
        update_exp_result(self.overall_log, task_id, bwf=BWF.avg)
        if task_id == 1 and self.config.task_prototype:
            self._duplicate_weights()
            self._save_checkpoint(0, save_best=True)
        return BWF.avg

    def _load_checkpoint(self, model_name, task=None):
        """
        Load from saved checkpoints
        :param model_name: Model name experiment to be loaded
        """
        if task is not None:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"Task{task}", model_name)
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, model_name)
        print("Loading checkpoint: {} ...".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        state_dict = checkpoint['state_dict']
        
        self.model.load_state_dict(state_dict)
        print("Checkpoint loaded")

        if 'list_val_acc_ii' in checkpoint:
            self.list_val_acc_ii = checkpoint['list_val_acc_ii']

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param save_best: if True, save checkpoint to 'model_best.pth'
        """
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'list_val_acc_ii': self.list_val_acc_ii,
        }

        save_path = self.checkpoint_dir

        if save_best:
            best_path = os.path.join(save_path, f'task{self.task_id}_model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")
        else:
            save_dir = os.path.join(save_path, 'backup')
            os.makedirs(save_dir, exist_ok=True)
            filename = os.path.join(save_dir, f'checkpoint-task-{self.task_id}-epoch-{epoch}.pth')
            torch.save(state, filename)
            print("Saving checkpoint: {} ...".format(filename))

    def _duplicate_weights(self):
        """
        Duplicate weights across experts (Only for frame_fusion_moe)
        """
        for i in range(len(self.model.clipmodel.text_model.encoder.layers)):
            for proj in ['q', 'k', 'v', 'out']:
                # Get source weights from first expert
                source_weights = getattr(self.model.clipmodel.text_model.encoder.layers[i].self_attn, f"{proj}_lora").lora_Bs[0].weight.data
                # Copy weights to all other experts
                with torch.no_grad():
                    for j in range(1, len(getattr(self.model.clipmodel.text_model.encoder.layers[i].self_attn, f"{proj}_lora").lora_Bs)):
                        getattr(self.model.clipmodel.text_model.encoder.layers[i].self_attn, f"{proj}_lora").lora_Bs[j].weight.data.copy_(source_weights)

        print("Successfully duplicated weights from first expert to all other experts")
