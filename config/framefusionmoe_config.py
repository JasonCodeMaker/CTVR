from config.base_config import BaseConfig


class FrameFusionMoEConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        
        self.add_args()        
        self.parse_args()

    def add_args(self):
        """Define custom arguments for FrameFusionMoE"""
        # Add new arguments to the parser
        # FFA
        self.parser.add_argument('--adapter_applied_layer', type=int, default=10, help="Number of layers to apply FFA")
        # TAME
        self.parser.add_argument('--lora_r', type=int, default=64, help="Rank of LoRA matrices")
        self.parser.add_argument('--lora_alpha', type=int, default=256, help="Alpha value for LoRA")
        self.parser.add_argument('--lora_nums', type=int, default=10, help="Number of LoRA experts")
        self.parser.add_argument('--lora_dropout', type=float, default=0.1, help="Dropout for LoRA")
        self.parser.add_argument('--topk', type=int, default=2, help="Top-k Selection")
        # Task Prototype
        self.parser.add_argument('--task_num', type=int, default=10, help="the num of tasks")
        self.parser.add_argument('--task_prototype', action='store_true', default=False, help="Use task prototype")
        # Cross-Task Loss   
        self.parser.add_argument('--loss_scale', type=float, default=0.6, help="Scale the loss")

        # Override defaults for existing parameters
        self.parser.set_defaults(
            clip_v_lr=3e-6,
            clip_t_lr=3e-6,
            noclip_lr=1e-5,
            num_frames=12, 
            batch_size=8,
            project_name='1-MSRVTT_10_FrameFusionMoE',
            dataset_name='MSRVTT',
            videos_dir='datasets/msrvtt_data/MSRVTT_Frames',
            benchmark = 'cap',
            arch='frame_fusion_moe',
            loss='triplet',
            grad_acc_steps=1,
            max_num_epochs=1,
            evals_per_epoch=1,
            # add other parameter overrides here
        )