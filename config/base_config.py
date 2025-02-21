import os
import argparse
import ast
from modules.basic_utils import mkdirp

class BaseConfig:
    def __init__(self):
        self.args = None
        self.parser = self.create_parser()

    def create_parser(self):
        '''Create an argument parser for the general configuration'''
        description = 'Continual Text-to-Video Retrieval'
        parser = argparse.ArgumentParser(description=description)
        
        # evaluation parameters
        parser.add_argument('--eval', action='store_true', default=False, help="Evaluation mode")
        parser.add_argument('--eval_path', type=str, default='outputs/debug', help="Path to the model to evaluate")

        # data parameters
        parser.add_argument('--path_data', type=str, default='data/MSRVTT_10_dataset.pkl', help="Path to sequential data file")
        parser.add_argument('--dataset_name', type=str, default='MSRVTT', help="Dataset name")
        parser.add_argument('--videos_dir', type=str, default='datasets/msrvtt_data/MSRVTT_Videos', help="Location of videos")
        parser.add_argument('--videos_train_dir', type=str, default='datasets/ActivityNet/Activity_Clip_Frames/dataset5', help="Location of training videos")
        parser.add_argument('--videos_val_dir', type=str, default='datasets/ActivityNet/Activity_Clip_Frames/val', help="Location of validation videos")
        parser.add_argument('--num_frames', type=int, default=12)
        parser.add_argument('--video_sample_type', default='uniform', help="'rand'/'uniform'")
        parser.add_argument('--input_res', type=int, default=224)

        # experiment parameters
        parser.add_argument('--project_name', type=str, default='test', help="Comet project name")
        parser.add_argument('--api_key', type=str, default='YOUR API KEY', help="Comet API key")
        parser.add_argument('--workspace', type=str, default='YOUR WORKSPACE', help="Comet workspace")
        parser.add_argument('--exp_name', type=str, default='debug', help="Name of the current experiment")
        parser.add_argument('--output_dir', type=str, default='./outputs')
        parser.add_argument('--evals_per_epoch', type=int, default=10, help="Number of times to evaluate per epoch")
        parser.add_argument('--load_epoch', type=int, help="Epoch to load from exp_name, or -1 to load model_best.pth")
        parser.add_argument('--eval_window_size', type=int, default=5, help="Size of window to average metrics")
        parser.add_argument('--metric', type=str, default='t2v', help="'t2v'/'v2t'")
        parser.add_argument('--grad_acc_steps', type=int, default=4, help="Number of steps to accumulate gradients")
        
        # continual learning settings
        parser.add_argument('--training_type', type=str, default='full_shot', help="Full shot or few shot")
        parser.add_argument('--num_shots', type=int, default=16, help="Number of shots for few-shot learning")
        parser.add_argument('--init_validation', action='store_true', default=True, help="Whether to validate before training")
        parser.add_argument('--store_vid_embed', action='store_true', default=False, help="Whether to store video embeddings from previous tasks")
        parser.add_argument('--load_best', action='store_true', default=True, help="Load the best model of each task for the final validation")
        parser.add_argument('--benchmark', type=str, default='anet_cap', help="Benchmark to use")
        parser.add_argument('--image_feature', action='store_true', default=False, help="Use additional image feature")
        parser.add_argument('--frame_reduction', type=int, default=-1, help="number of remaining frames")
        parser.add_argument('--wise_alpha', type=float, default=0.5, help="Alpha value for WISE-FT")

        # WISE-FT parameters
        parser.add_argument('--wise_mode', type=str, default='None', help="Which mode to use for WISE-FT")
        parser.add_argument('--wise_strategy', type=str, default='interpolation', help="Which strategy to use for WISE-FT")

        # model parameters
        parser.add_argument('--huggingface', action='store_true', default=True)
        parser.add_argument('--pre_trained', action='store_false', default=True, help="Use pre-trained CLIP model")
        parser.add_argument('--arch', type=str, default='clip_transformer')
        parser.add_argument('--clip_arch', type=str, default='ViT-B/32', help="CLIP arch. only when not using huggingface")
        parser.add_argument('--embed_dim', type=int, default=512, help="Dimensionality of the model embedding")

        # training parameters
        parser.add_argument('--loss', type=str, default='clip')
        parser.add_argument('--clip_v_lr', type=float, default=1e-6, help='Learning rate used for CLIP vision params')
        parser.add_argument('--clip_t_lr', type=float, default=1e-6, help='Learning rate used for CLIP text params')
        parser.add_argument('--noclip_lr', type=float, default=1e-5, help='Learning rate used for new params')
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--max_num_epochs', type=int, default=3)
        parser.add_argument('--weight_decay', type=float, default=0.2, help='Weight decay')
        parser.add_argument('--warmup_proportion', type=float, default=0.1, help='Warmup proportion for learning rate schedule')

        # frame pooling parameters
        parser.add_argument('--pooling_type', type=str)
        parser.add_argument('--attention_temperature', type=float, default=0.01, help='Temperature for softmax (used in attention pooling only)')
        parser.add_argument('--num_mha_heads', type=int, default=1, help='Number of parallel heads in multi-headed attention')
        parser.add_argument('--transformer_dropout', type=float, default=0.3, help='Dropout prob. in the transformer pooling')

        # system parameters
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--seed', type=int, default=42, help='Random seed')
        parser.add_argument('--data_analysis', action='store_true', default=False, help='Data analysis mode')

        # MoE Adapter
        parser.add_argument("--ffn_num", type=int, default=64, help="the dim of lora")
        parser.add_argument("--ffn_adapt", action="store_true", help="Whether or not to use adapter")  # use adapter
        parser.add_argument("--ffn_option", type=str, default="parallel")
        parser.add_argument("--ffn_adapt_where", type=str,
            default="AdapterDoubleEncoder",
            choices=["AdapterImageEncoder", "AdapterDoubleEncoder"],
            help="Adapter is added into ImageEncoder or ImageEncoder_and_TextEncoder.",)  # not use
        parser.add_argument("--apply_moe", action="store_true", help="Whether or not to use moe")  # use moe
        parser.add_argument("--repeat_train", action="store_true", help="default is manual differentiation")
        parser.add_argument("--task_id", type=int, default=-1, help="task_id")
        parser.add_argument("--multi_experts", action="store_true", help="Whether or not to use multi_experts")
        parser.add_argument("--num_experts", type=int, default=22, help="the number of experts")
        parser.add_argument("--is_train", action="store_true", help="Whether or not to use router noise")
        parser.add_argument("--frozen", action="store_true", help="Whether or not to use adapter")
        parser.add_argument("--autorouter", action="store_true", help="Whether or not to use autorouter")
        parser.add_argument("--threshold", type=float, help="threshold for zero-shot.")
        parser.add_argument("--non_text", action="store_true", help="不使用text——encoder测试")
        parser.add_argument("--frozen_path", type=str, default="frozen_list")

        return parser

    def parse_args(self):
        """Actually parse the arguments"""
        if self.args is None:
            self.args = self.parser.parse_args()
            self.args.model_path = os.path.join(self.args.output_dir, self.args.exp_name)
            mkdirp(self.args.model_path)
            
        self._copy_args_to_attributes()
        return self.args
    
    def _copy_args_to_attributes(self):
        """Copy parsed arguments to instance attributes"""
        for key, value in vars(self.args).items():
            setattr(self, key, value)

    def print_config(self):
        """Print all configuration parameters."""
        print("Configuration Parameters:")
        print("=" * 30)
        for attr, value in sorted(vars(self.args).items()):
            print(f"{attr.ljust(30)}: {value}")
        print("=" * 30)