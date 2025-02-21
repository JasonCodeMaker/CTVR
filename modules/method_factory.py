class MethodFactory:
    @staticmethod
    def get_trainer(config):
        if config.arch == 'frame_fusion_moe':
            from trainer.trainer_acc_framefusionMoE import Trainer as FrameFusionTrainer
            return FrameFusionTrainer
        # Add additional methods here as elif branches.
        # elif config.arch == 'other_method':
        #     from trainer.trainer_acc_other import Trainer as OtherTrainer
        #     return OtherTrainer
        else:
            raise NotImplementedError(f"Trainer for '{config.arch}' is not implemented.")

    @staticmethod
    def get_evaluator(config):
        if config.arch == 'frame_fusion_moe':
            from trainer.trainer_acc_framefusionMoE import Evaluator as FrameFusionEvaluator
            return FrameFusionEvaluator
        # Add additional methods here as elif branches.
        # elif config.arch == 'other_method':
        #     from trainer.trainer_acc_other import Evaluator as OtherEvaluator
        #     return OtherEvaluator
        else:
            raise NotImplementedError(f"Evaluator for '{config.arch}' is not implemented.")
