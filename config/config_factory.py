import argparse
from config.base_config import BaseConfig

def get_arch_from_argv():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--arch', type=str, default='frame_fusion_moe', help="method name")
    args, _ = parser.parse_known_args()
    return args.arch


from config.framefusionmoe_config import FrameFusionMoEConfig
# from config.avg_pool_config import AvgPoolConfig
# from config.xpool_config import XPoolConfig
# from config.moe_adapter_config import MoEAdapterConfig
# from config.clip_vip_config import ClipVipConfig

class ConfigFactory:
    @staticmethod
    def get_config() -> BaseConfig:
        arch = get_arch_from_argv()
        if arch =='frame_fusion_moe':
            return FrameFusionMoEConfig()
        # elif arch == 'avg_pool':
        #     return AvgPoolConfig()
        # elif arch == 'xpool':
        #     return XPoolConfig()
        # elif arch == 'moe_adapter':
        #     return MoEAdapterConfig()
        # elif arch == 'clip_vip':
        #     return ClipVipConfig()
        else:
            raise NotImplementedError(f"Configuration for architecture '{arch}' is not implemented.")
