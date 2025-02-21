from config.config_factory import ConfigFactory
from model.model_factory import ModelFactory
from modules.basic_utils import seed_everything
from datasets.cil_dataset import CILSetTask
from modules.common import initialize_experiment, load_dataset
from modules.method_factory import MethodFactory
from trainer.trainer_framework import run_training, run_evaluation


def main():
    # 1. Load configuration
    config = ConfigFactory.get_config()
    config.print_config()
    
    # 2. Initialize CometML experiment
    experiment = initialize_experiment(config)
    experiment.set_name(config.exp_name)
    
    # 3. Set seed
    seed = seed_everything(config.seed)
    print(f"Seed: {seed}")
    
    # 4. Load dataset
    data = load_dataset(config.path_data)
    
    # 5. Initialize model(s)
    model = ModelFactory.get_model(config)
    ref_model = ModelFactory.get_model(config)
    
    # 6. Prepare continual learning dataset handlers
    train_cil = CILSetTask(data['train'], model, 'train', config.dataset_name, config)
    val_cil = CILSetTask(data['val'], model, 'test', config.dataset_name, config)
    
    # 7. Execute Training or Evaluation
    if config.eval:
        evaluator_cls = MethodFactory.get_evaluator(config)
        run_evaluation(config, model, val_cil, experiment, evaluator_cls)
    else:
        trainer_cls = MethodFactory.get_trainer(config)
        current_task = 2
        run_training(current_task, config, ref_model, model, train_cil, val_cil, experiment, trainer_cls)

if __name__ == '__main__':
    main()
