import os
from accelerate import Accelerator
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import defaultdict, deque
import time

from trainer.base_trainer import BaseTrainer
from modules.optimization import AdamW
from datasets.utils.cached_video_dataset import ReferenceVideoDataset, RefDataIterator
from modules.trainer_utils import log_training_progress, load_stored_embed, update_exp_result
from evaluator.validator import Validator


class Trainer(BaseTrainer):
    def __init__(self, ref_model, model, loss, metrics, current_task_id, config, 
                 train_data_loader, valid_data_loader, tokenizer, list_val_acc_ii, 
                 num_epochs, experiment=None):
        """
        Initialize the Trainer and configure model-specific settings.
        """
        # Call the parent constructor (BaseTrainer handles generic setup)
        super().__init__(model, loss, metrics, current_task_id, num_epochs, config)
        
        self.current_task_id = current_task_id
        self.config = config
        self.num_epochs = num_epochs
        self.experiment = experiment
        self.tokenizer = tokenizer
        self.list_val_acc_ii = list_val_acc_ii
        self.best = -1.0
        self.task_best = -1.0
        self.step = 0

        # Initialize Accelerator and set gradient accumulation steps
        self._initialize_accelerator()
        
        # Prepare the reference model (set to eval mode)
        self._prepare_reference_model(ref_model)
        
        # Assign data loaders
        self.train_data_loader = train_data_loader
        self.val_loaders_list = valid_data_loader
        
        # Freeze model parameters (model-specific freezing strategy)
        self._freeze_model_parameters()
        
        # Configure optimizer by grouping parameters and setting learning rates
        self._configure_optimizer()
        
        # Prepare the model, optimizer, and training data loader with Accelerator
        self._prepare_with_accelerator()
        
        # Set up logging files (overall and per-task)
        self._setup_logging()
        
        # Reset model-specific counters (e.g., LoRA counters)
        self._reset_model_counters()
        
        # If using 'triplet' loss and current task > 1, load previous video embeddings
        if self.config.loss == 'triplet' and self.current_task_id > 1:
            self._setup_triplet_loss()

        # Construct the Validator object for validation 
        self.validator = Validator(
            self.model, metrics, config, valid_data_loader, tokenizer, 
            self.accelerator, experiment, self.checkpoint_dir, self.list_val_acc_ii,
            self.task_log, self.overall_log
        )

    def _freeze_model_parameters(self):
        """
        Apply the freezing strategy for model parameters based on their names 
        and the current task.
        """
        for name, param in self.model.named_parameters():
            if "vision_model" in name:
                param.requires_grad = False
                if "frame_cross_attention" in name or "alpha" in name:
                    param.requires_grad = True
            elif "text_model" in name:
                param.requires_grad = False
                if self.current_task_id == 1:
                    if "lora_A" in name or "lora_Bs.0" in name or "w_noise" in name or "task_prototype" in name:
                        param.requires_grad = True
                else:
                    if "lora" in name or "w_noise" in name or "task_prototype" in name:
                        param.requires_grad = True

    def _configure_optimizer(self):
        """
        Group model parameters and configure the optimizer with different learning
        rates based on parameter type and current task.
        """
        params_optimizer = list(self.model.named_parameters())
        
        self.clip_text_params = [
            p for n, p in params_optimizer 
            if "text_model" in n and ("lora" in n or "w_noise" in n) and p.requires_grad
        ]
        self.clip_vision_params = [
            p for n, p in params_optimizer 
            if "frame_cross_attention" in n and p.requires_grad
        ]
        self.noclip_params = [
            p for n, p in params_optimizer 
            if (("alpha" in n or "task_prototype" in n) and p.requires_grad)
        ]
        
        # Set learning rates based on current task and dataset
        if self.current_task_id == 1:
            if self.config.dataset_name == 'MSRVTT':
                clip_t_lr = 2e-5
                clip_v_lr = 2e-5
                noclip_lr = 2e-5
            elif self.config.dataset_name == 'ACTNET':
                clip_t_lr = 5e-5
                clip_v_lr = 5e-5
                noclip_lr = 1e-4
            else:
                clip_t_lr = 2e-5
                clip_v_lr = 2e-5
                noclip_lr = 2e-5
        else:
            clip_t_lr = float(self.config.clip_t_lr)
            clip_v_lr = float(self.config.clip_v_lr)
            noclip_lr = float(self.config.noclip_lr)
        
        print(f"clip_text_LR: {clip_t_lr}, clip_vision_LR: {clip_v_lr}, noclip_LR: {noclip_lr}")
        
        optimizer_grouped_params = [
            {'params': self.clip_text_params, 'lr': clip_t_lr, 'name': 'clip_text'},
            {'params': self.clip_vision_params, 'lr': clip_v_lr, 'name': 'clip_vision'},
            {'params': self.noclip_params, 'lr': noclip_lr, 'name': 'non_clip'}
        ]
        
        total_params_size = sum(p.numel() * p.element_size() for p in self.model.parameters() if p.requires_grad)
        print('The number of Total Trainable Parameters:', sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        # print(f"Total Trainable Parameters Memory Size: {total_params_size / 1024 / 1024:.2f} MB")
        
        self.optimizer = AdamW(optimizer_grouped_params, weight_decay=self.config.weight_decay)

    def _reset_model_counters(self):
        """
        Reset all LoRA counters in the model (if applicable).
        """
        self.model.reset_all_lora_counters()

    def _setup_triplet_loss(self):
        """
        If using 'triplet' loss and current task > 1, load video embeddings from previous tasks.
        """
        self.ref_vid_embeds = []
        for n_task in range(self.current_task_id - 1):
            task_vid_embeds = load_stored_embed(self.checkpoint_dir, n_task + 1)
            self.ref_vid_embeds.append(task_vid_embeds.to(self.accelerator.device))
        
        cached_dataset = ReferenceVideoDataset(self.ref_vid_embeds)
        ref_vid_loader = DataLoader(
            cached_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        self.ref_vid_loader = RefDataIterator(ref_vid_loader, self.device)

    def set_scheduler(self, scheduler):
        self.lr_scheduler = self.accelerator.prepare(scheduler)

    def get_list_val_acc_ii(self):
        return self.list_val_acc_ii

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.
        """
        if epoch == 1:
            if self.config.init_validation:
                self.validator.task_validation(self.task_id, 0)
            print("Starting Training...")

        self.model.train()
        total_loss = 0.0
        num_steps = len(self.train_data_loader)
        
        epoch_start_time = time.time()
        with self.experiment.train():
            for batch_idx, data in enumerate(self.train_data_loader):
                # then assume we must tokenize the input, e.g. its a string
                if self.tokenizer is not None:
                    data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True,
                                                truncation=True)
                if isinstance(data['text'], torch.Tensor):
                    data['text'] = data['text'].to(self.accelerator.device)
                else:
                    data['text'] = {key: val.to(self.accelerator.device) for key, val in data['text'].items()}
                
                data['video'] = data['video'].to(self.accelerator.device)
                if self.config.task_prototype:
                    data['prototype_id'] = self.task_id

                with self.accelerator.autocast():
                    text_embeds, video_embeds = self.model(data, image=False)

                    if self.config.loss == 'NCELearnableTempLoss':
                        loss = self.loss(video_embeds, text_embeds, self.model.clipmodel.logit_scale)
                    elif self.config.loss == 'lwf':
                        with torch.no_grad():
                            ref_text_embeds, ref_video_embeds = self.ref_model(data)
                        loss = self.loss(video_embeds, text_embeds, self.model.clipmodel.logit_scale, ref_video_embeds, ref_text_embeds)
                    elif self.config.loss == 'triplet':
                        if self.task_id > 1:
                            ref_vid_embeds = self.ref_vid_loader.get_next()
                        else:
                            ref_vid_embeds = None    
                        loss = self.loss(video_embeds, text_embeds, self.model.clipmodel.logit_scale, ref_vid_embeds, self.config.loss_scale)
                    else:
                        raise NotImplementedError(f"Loss {self.config.loss} not implemented")

                    scaled_loss = loss / self.gradient_accumulation_steps

                self.accelerator.backward(scaled_loss)
                # self.model.get_gradient_stats()

                # Log the current loss and learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                total_loss += loss.detach().item() 

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or batch_idx == len(self.train_data_loader) - 1:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    torch.clamp_(self.model.clipmodel.logit_scale.data, 0, np.log(200))
                    self.step += 1

                    # Print progress information
                    log_training_progress(self.experiment, self.task_id, self.step, epoch, batch_idx, num_steps, 
                                        epoch_start_time, loss.detach().item(), current_lr)

        # Evaluate on the validation set 
        if epoch % self.evals_per_epoch == 0 or epoch == self.total_epochs:
            task_res = self.validator.task_validation(self.task_id, epoch)
            
            if self.config.load_best:
                # First get the overall validation result
                if self.task_id > 1:
                    overall_res = self.validator.validate(self.task_id, epoch, self.step)
                    current_score = overall_res['R1']
                else:
                    # For first task, overall performance equals task performance
                    overall_res = task_res
                    current_score = task_res['R1']
                    
                # Compare using overall score instead of task score
                if current_score >= self.best:
                    self.best = current_score  # Update best with overall score
                    self._save_checkpoint(epoch, save_best=True)
                    self.global_best = overall_res['R1']
                    
                    # Log results
                    update_exp_result(self.task_log, self.task_id, 
                                    r1=task_res['R1'], r5=task_res['R5'], 
                                    r10=task_res['R10'], medr=task_res['MedR'], 
                                    meanr=task_res['MeanR'])
                    update_exp_result(self.overall_log, self.task_id, 
                                    r1=overall_res['R1'], r5=overall_res['R5'], 
                                    r10=overall_res['R10'], medr=overall_res['MedR'], 
                                    meanr=overall_res['MeanR'])
                    self.experiment.log_metric("CIL_Performance(R@1)", overall_res['R1'], step=self.task_id)
                    self.experiment.log_metric("CIL_Performance(R@5)", overall_res['R5'], step=self.task_id)

                    print(f"\nCurrent Best Overall R@1 is {self.best:.6f}")
                else:
                    print(f"\nCurrent Overall R@1 is {current_score:.6f} < {self.best:.6f}")
            else:
                self.best = task_res['R1']
                print(f"\nCurrent Final R@1 is {self.best:.6f}")

        res = {
            'loss_train':  total_loss / num_steps
        }
            
        # memory_callback() # Debugging
        torch.cuda.empty_cache()
        return res
        
        
class Evaluator(Trainer):
    def __init__(self, model, metrics, config, valid_data_loader, tokenizer, 
                 list_val_acc_ii, experiment=None):
        self.model = model
        self.metrics = metrics
        self.config = config
        self.val_loaders_list = valid_data_loader
        self.tokenizer = tokenizer
        self.list_val_acc_ii = list_val_acc_ii
        self.experiment = experiment
        
        # Initialize accelerator
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        
        # Prepare model
        self.model = self.accelerator.prepare(self.model)
        
        # Initialize other necessary attributes from parent class
        self.window_metric = defaultdict(list)
        self.checkpoint_dir = config.eval_path

        # Construct the Validator object for validation 
        self.validator = Validator(
            self.model, metrics, config, valid_data_loader, tokenizer, 
            self.accelerator, experiment, self.checkpoint_dir,
            self.task_log, self.overall_log
        )