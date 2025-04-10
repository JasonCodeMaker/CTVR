class ModelFactory:
    @staticmethod
    def get_model(config):
        if config.arch == 'avg_pool':
            from model.AvgPool.AvgPool import AvgPool
            return AvgPool(config)
        elif config.arch == 'xpool':
            from model.XPool.clip_transformer import XPool
            return XPool(config)
        elif config.arch == 'moe_adapter':
            from model.MoE_Adapter.MoEAdapter import MoEAdapter
            return MoEAdapter(config)
        elif config.arch == 'clip_vip':
            from model.CLIP_ViP.ClipViP import ClipVip
            return ClipVip(config)
        elif config.arch == 'frame_fusion_moe':
            from model.FrameFusionMoE.FrameFusion import FrameFusion
            return FrameFusion(config)
        elif config.arch == 'frame_fusion_moe_eval':
            from model.FrameFusionMoE_Eval.FrameFusion import FrameFusion
            return FrameFusion(config)
        else:
            raise NotImplemented
