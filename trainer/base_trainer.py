import torch
import os
from abc import abstractmethod
from accelerate import Accelerator
from modules.trainer_utils import construct_exp_log

class BaseTrainer:
    """
    Base class for all trainers.
    """
    def __init__(self, model, loss, metrics, current_task_id, num_epochs, config):
        """
        Initialize the base trainer.

        Args:
            model: The model to be trained.
            loss: The loss function.
            metrics: Evaluation metrics.
            current_task_id: The current task identifier.
            num_epochs: The number of training epochs.
            config: A configuration object (BaseConfig) containing training parameters.
        """
        self.config = config
        # Set up GPU device if available and move the model to the device.
        self.device = self._prepare_device()
        self.model = model.to(self.device)

        self.task_id = current_task_id
        self.total_epochs = num_epochs + 1  # Total number of epochs (including initial state)

        self.loss = loss.to(self.device)
        self.metrics = metrics

        self.start_epoch = 1
        self.global_step = 0

        self.checkpoint_dir = config.model_path
        # Initialize frozen_path only if the freezing strategy is enabled in config.
        if hasattr(config, 'frozen') and config.frozen:
            self.frozen_path = os.path.join(self.checkpoint_dir, "frozen_list_experts.txt")

        self.evals_per_epoch = config.evals_per_epoch

    def _initialize_accelerator(self):
        """Initialize Accelerator and set gradient accumulation steps."""
        self.accelerator = Accelerator()
        self.gradient_accumulation_steps = self.config.grad_acc_steps

    def _prepare_reference_model(self, ref_model):
        """Prepare the reference model and set it to evaluation mode."""
        self.ref_model = ref_model.to(self.device)
        self.ref_model.eval()


    def _prepare_with_accelerator(self):
        """
        Use Accelerator to prepare the reference model, model, optimizer, and training data loader.
        """
        self.ref_model, self.model, self.optimizer, self.train_data_loader = self.accelerator.prepare(
            self.ref_model, self.model, self.optimizer, self.train_data_loader
        )

    def _setup_logging(self):
        """
        Set up log file paths and initialize log files if they do not exist.
        """
        self.overall_log = os.path.join(self.checkpoint_dir, "overall_log.csv")
        if not os.path.exists(self.overall_log):
            construct_exp_log(self.overall_log, num_tasks=20)
        self.task_log = os.path.join(self.checkpoint_dir, "task_log.csv")
        if not os.path.exists(self.task_log):
            construct_exp_log(self.task_log, num_tasks=20)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self, memory_callback=None):
        """
        Full training logic with support for frozen modules
        """
        for epoch in range(self.start_epoch, self.total_epochs):
            self._train_epoch(epoch)

        # Handle frozen modules if configured
        if hasattr(self.config, 'frozen') and self.config.frozen:
            self._handle_frozen_modules()

    def _handle_frozen_modules(self):
        """
        Handles the frozen modules logic previously in train()
        """
        if self.task_id > 1:
            with open(self.frozen_path, "a") as file:
                for i in range(12):
                    for proj in ['q', 'k', 'v', 'out']:
                        choose_map = getattr(self.model.clipmodel.text_model.encoder.layers[i].self_attn, f"{proj}_lora").choose_map
                        top_values_t, top_indices_t = torch.topk(choose_map, self.config.topk-1)

                        for k in range(len(top_indices_t)):
                            item1 = 'clipmodel.text_model.encoder.layers.{}.self_attn.{}_lora.lora_Bs.{}.weight'.format(i, proj, top_indices_t[k])
                            file.write(item1 + "\n")
        else:
            with open(self.frozen_path, "w") as file:
                for i in range(12):
                    for proj in ['q', 'k', 'v', 'out']:
                        choose_map = getattr(self.model.clipmodel.text_model.encoder.layers[i].self_attn, f"{proj}_lora").choose_map
                        item1 = 'clipmodel.text_model.encoder.layers.{}.self_attn.{}_lora.lora_Bs.0.weight'.format(i, proj)
                        file.write(item1 + "\n")

    def _prepare_device(self):
        """
        setup GPU device if available, move model into configured device
        """
        use_gpu = torch.cuda.is_available()
        device = torch.device('cuda:0' if use_gpu else 'cpu')
        return device

    def _save_checkpoint(self, epoch, save_best=False, task_id=None, lr=None):
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

        # Add route_weights if they exist
        if hasattr(self, 'route_weights_ckpts'):
            state['route_weights'] = self.route_weights_ckpts

        if task_id is not None:
            save_path = os.path.join(self.checkpoint_dir, f"Task{self.task_id}")
        else:
            save_path = self.checkpoint_dir

        if save_best:
            if lr is not None:
                best_path = os.path.join(save_path, f'task{self.task_id}_model_best_lr_{lr}.pth')
            else:
                best_path = os.path.join(save_path, f'task{self.task_id}_model_best.pth')

            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")
        else:
            save_dir = os.path.join(save_path, 'backup')
            os.makedirs(save_dir, exist_ok=True)
            filename = os.path.join(save_dir, f'checkpoint-task-{self.task_id}-epoch-{epoch}.pth')
            torch.save(state, filename)
            print("Saving checkpoint: {} ...".format(filename))

    def save_checkpoint(self, model_name):
        """
        Save checkpoint with a specific name
        """
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'list_val_acc_ii': self.list_val_acc_ii,
        }

        best_path = os.path.join(self.checkpoint_dir, model_name)
        torch.save(state, best_path)
        print("Saving current global best: model_global_best.pth ...")

    def _save_pretrained_model(self):
        """
        Saving pretrained model checkpoints
        """
        state = {
            'epoch': 0,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'list_val_acc_ii': self.list_val_acc_ii,
        }

        save_path = self.checkpoint_dir
        best_path = os.path.join(save_path, 'task0_model_best.pth')
        torch.save(state, best_path)

    def load_checkpoint(self, model_name, task=None):
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

    def _load_state_dict(self, model_name, task=None):
        """
        Load and return state dict from checkpoint
        """
        if task is not None:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"Task{task}", model_name)
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, model_name)

        checkpoint = torch.load(checkpoint_path, weights_only=True)
        return checkpoint['state_dict']

    def save_text_embeddings(self, text_embeds, n_task, save_dir):
        """
        Save text embeddings for a specific task
        """
        os.makedirs(save_dir, exist_ok=True)
        filename = f"text_embeddings_task_{n_task}.pt"
        save_path = os.path.join(save_dir, filename)
        torch.save(text_embeds, save_path)
        print(f"Text embeddings for task {n_task} saved to {save_path}")
