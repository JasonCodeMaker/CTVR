from config.base_config import BaseConfig
from torch.utils.data import DataLoader

class DataFactory:

    @staticmethod
    def get_data_loader(config, dataset_name, data, model, split_type, img_transforms, replayed_pairs=None):

        if dataset_name == "MSRVTT":
            if config.frame_reduction > -1:
                from datasets.msrvtt_dataset_f_select import MSRVTTDataset
            else:
                from datasets.msrvtt_dataset import MSRVTTDataset

            if split_type == 'train':
                dataset = MSRVTTDataset(config, data, model, split_type, img_transforms, replayed_pairs)

                return DataLoader(dataset, batch_size=config.batch_size,
                           shuffle=True, num_workers=config.num_workers, pin_memory = True), dataset
            else:
                dataset = MSRVTTDataset(config, data, model, split_type, img_transforms)
                return DataLoader(dataset, batch_size=config.batch_size,
                           shuffle=False, num_workers=config.num_workers, pin_memory = True)
        
        elif dataset_name == 'ACTNET':
            if config.frame_reduction > -1:
                from datasets.actnet_dataset_f_select import ACTNETDataset
            else:
                from datasets.actnet_dataset import ACTNETDataset

            if split_type == 'train':
                dataset = ACTNETDataset(config, data, model, split_type, img_transforms)
                return DataLoader(dataset, batch_size=config.batch_size,
                            shuffle=True, num_workers=config.num_workers, pin_memory = True), dataset
            else:
                dataset = ACTNETDataset(config, data, model, split_type, img_transforms)
                return DataLoader(dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=config.num_workers, pin_memory = True)

        else:
            raise NotImplementedError
