import math
import os
import torch
from modules.loss import LossFactory
from modules.metrics import t2v_metrics, v2t_metrics
from modules.optimization import get_cosine_schedule_with_warmup
from modules.common import get_tokenizer

def train_task(task_id, model, ref_model, train_loader, valid_loader, metrics, config, experiment, trainer_cls, list_val_acc_ii, tokenizer):
    """
    Execute the full training workflow for a single task.

    """
    # Initialize the trainer
    trainer = trainer_cls(
        model=model,
        ref_model=ref_model,
        train_data_loader=train_loader,
        valid_data_loader=valid_loader,
        loss=LossFactory.get_loss(config),
        tokenizer=tokenizer,
        list_val_acc_ii=list_val_acc_ii,
        num_epochs=config.max_num_epochs,
        metrics=metrics,
        current_task_id=task_id,
        config=config,
        experiment=experiment,
    )
    
    # For tasks beyond the first, load the best model from the previous task
    if task_id > 1:
        prev_ckpt = f"task{task_id-1}_model_best.pth"
        trainer.load_checkpoint(prev_ckpt)
    
    # Configure the learning rate scheduler (this part is generic)
    gradient_accumulation_steps = config.grad_acc_steps
    num_update_steps_per_epoch = math.ceil(len(train_loader) / gradient_accumulation_steps)
    num_epochs = config.max_num_epochs
    num_training_steps = num_update_steps_per_epoch * num_epochs
    num_warmup_steps = int(config.warmup_proportion * num_training_steps) if task_id == 1 else 0

    scheduler = get_cosine_schedule_with_warmup(
        trainer.optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=1.0
    )
    trainer.set_scheduler(scheduler)
    
    # Start training
    trainer.train()
    
    # Evaluation phase (using trainer's evaluation methods)
    with experiment.validate():
        final_r1 = trainer.best
        BWF = trainer.validator.final_validate(task_id, trainer.get_list_val_acc_ii())
        experiment.log_metric("Final_R@1_Per_task", final_r1, step=task_id)
        experiment.log_metric("Final_BWF_Per_task", BWF, step=task_id)
        print(f"\nFinal R@1: {final_r1}")
        print(f"Final BWF: {BWF}")
    

def run_training(current_task, config, ref_model, model, train_data, val_data, experiment, trainer_cls):
    """
    Execute the full training workflow across tasks.
    """
    tokenizer = get_tokenizer(config)
    iter_trainDataloader = iter(train_data)
    num_tasks = iter_trainDataloader.num_tasks
    list_val_acc_ii = []

    if current_task > 1:
        for _ in range(1, current_task):
            next(iter_trainDataloader)
    
    for task_id in range(current_task, num_tasks):
        print(f"\n{'='*30} Training Task {task_id} {'='*30}")
        _, train_loader, _ = next(iter_trainDataloader)
        print(f"Current Training Classes: {list(train_loader.dataset.data.keys())}")
        val_loader = val_data.get_valSet_by_taskNum(task_id+1)
        

        # Define the metrics for the current task
        if config.metric == 't2v':
            metrics = t2v_metrics
        elif config.metric == 'v2t':
            metrics = v2t_metrics
        else:
            raise NotImplemented

        # Train the current task
        train_task(
            task_id=task_id,
            model=model,
            ref_model=ref_model,
            train_loader=train_loader,
            valid_loader=val_loader,
            metrics=metrics,
            config=config,
            experiment=experiment,
            trainer_cls=trainer_cls,
            list_val_acc_ii=list_val_acc_ii,
            tokenizer=tokenizer
        )
        
        torch.cuda.empty_cache()

def run_evaluation(config, model, val_data, experiment, evaluator_cls):
    """Execute the evaluation workflow based on configuration."""
    tokenizer = get_tokenizer(config)
    
    # Get the total number of tasks
    total_tasks = val_data.num_tasks - 1  
    
    # Select the evaluation metric type
    if config.metric == 't2v':
        from modules.metrics import t2v_metrics
        metrics = t2v_metrics
    elif config.metric == 'v2t':
        from modules.metrics import v2t_metrics
        metrics = v2t_metrics
    else:
        raise NotImplementedError(f"Metric {config.metric} not implemented")
    
    # Initialize the list to store validation accuracy
    list_val_acc_ii = []
    
    # Determine which task ID to evaluate
    if not hasattr(config, 'eval_task_id') or config.eval_task_id is None:
        eval_task_id = total_tasks  # By default, evaluate the last task
    else:
        eval_task_id = min(config.eval_task_id, total_tasks)  # Prevent out-of-range errors
    
    # Set the evaluation mode
    eval_mode = getattr(config, 'eval_mode', 'single')
    
    # Set the base path for checkpoints
    checkpoint_base = config.eval_path if hasattr(config, 'eval_path') and config.eval_path else config.model_path
    
    if eval_mode == 'single':
        # Only evaluate the specified task
        tasks_to_evaluate = [eval_task_id]
    else:  # eval_mode == 'all'
        # Evaluate all tasks from 1 to eval_task_id
        tasks_to_evaluate = list(range(1, eval_task_id + 1))
    
    print(f"Evaluation mode: {eval_mode}")
    print(f"Tasks to evaluate: {tasks_to_evaluate}")
    print(f"Using checkpoint base path: {checkpoint_base}")
    
    results_by_task = {}
    
    # Loop through each task to evaluate
    for task_id in tasks_to_evaluate:
        if task_id < 1 or task_id > total_tasks:
            print(f"Warning: Task ID {task_id} is out of range (1-{total_tasks}), skipping...")
            continue
            
        print(f"\n{'='*30} Evaluating Task {task_id} {'='*30}")
        
        # Get the validation set for the current task
        val_loaders = val_data.get_valSet_by_taskNum(task_id + 1)
        
        # For this task, use the first task_id loaders
        val_loaders_list = val_loaders[:task_id] if val_loaders else []
        
        if not val_loaders_list:
            print(f"Warning: No validation data available for task {task_id}, skipping...")
            continue
        
        # Build the checkpoint path
        ckpt_path = os.path.join(checkpoint_base, f"task{task_id}_model_best.pth")
        if not os.path.exists(ckpt_path):
            print(f"Warning: Checkpoint not found at {ckpt_path}, skipping task {task_id}")
            continue
        
        # Initialize the evaluator with all required parameters
        evaluator = evaluator_cls(
            model=model,
            metrics=metrics,                     
            config=config,
            eval_task_id=task_id,
            valid_data_loader=val_loaders_list,
            tokenizer=tokenizer,
            list_val_acc_ii=list_val_acc_ii,      
            experiment=experiment
        )
        
        # Load the checkpoint
        checkpoint = torch.load(ckpt_path, weights_only=True)
        evaluator.model.load_state_dict(checkpoint['state_dict'])
        print(f"Checkpoint loaded directly from: {ckpt_path}")

        # If list_val_acc_ii exists in the checkpoint, load it for final_validate
        if 'list_val_acc_ii' in checkpoint:
            list_val_acc_ii = checkpoint['list_val_acc_ii']

        # Run the evaluation
        results = evaluator.validator.validate(task_id, 0, 0)
        results_by_task[task_id] = results

        # Log the results
        experiment.log_metrics({
            f"Task{task_id}_R1": results['R1'],
            f"Task{task_id}_R5": results['R5'],
            f"Task{task_id}_R10": results['R10']
        }, step=task_id)

        torch.cuda.empty_cache()

    # For BWF, call final_validate for each task_id
    per_task_bwf = {}
    if len(results_by_task) > 0:
        for task_id in results_by_task.keys():
            bwf = evaluator.validator.final_validate(task_id, list_val_acc_ii)
            per_task_bwf[task_id] = bwf

    # Build the summary table as a string
    summary_lines = []
    summary_lines.append("="*30 + " Evaluation Summary " + "="*30)
    summary_lines.append(f"{'Task ID':<10}{'R@1':<10}{'R@5':<10}{'R@10':<10}{'MedR':<10}{'MeanR':<15}{'BWF':<10}")
    summary_lines.append("-" * 80)
    for task_id, res in results_by_task.items():
        bwf_val = per_task_bwf.get(task_id, 0.0)
        summary_lines.append(f"{task_id:<10}{res['R1']:<10.2f}{res['R5']:<10.2f}{res['R10']:<10.2f}{res['MedR']:<10.2f}{res['MeanR']:<15.2f}{bwf_val:<10.4f}")
    summary_str = "\n".join(summary_lines)

    # Print the summary
    print("\n" + summary_str)

    # Build the config parameters string, similar to config.print_config()
    config_lines = []
    config_lines.append("\nConfiguration Parameters:")
    config_lines.append("=" * 30)
    config_dict = config.config if hasattr(config, "config") else (vars(config) if hasattr(config, '__dict__') else dict(config))
    for k, v in sorted(config_dict.items()):
        config_lines.append(f"{str(k).ljust(30)}: {v}")
    config_lines.append("=" * 30)
    config_str = "\n".join(config_lines)

    # Generate the log file name
    eval_path = getattr(config, 'eval_path', None) or getattr(config, 'model_path', None) or "eval"
    log_folder = os.path.basename(os.path.normpath(eval_path))
    log_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(log_dir, exist_ok=True)
    # Create the log file name based on eval_mode
    if eval_mode == "all":
        log_filename = f"{log_folder}_all.log"
    else:
        log_filename = f"{log_folder}_task{eval_task_id}.log"
    log_path = os.path.join(log_dir, log_filename)

    # Write the summary and config to the log file
    with open(log_path, "w") as f:
        f.write(summary_str)
        f.write("\n")
        f.write(config_str)
        f.write("\n")

    return results_by_task
