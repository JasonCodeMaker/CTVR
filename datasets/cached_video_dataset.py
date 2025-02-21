import torch
from torch.utils.data import Dataset


class ReferenceVideoDataset(Dataset):
    def __init__(self, ref_vid_embeds):
        self.all_embeds = torch.cat([embed.cpu() for embed in ref_vid_embeds], dim=0)
        
    def __len__(self):
        return len(self.all_embeds)
    
    def __getitem__(self, idx):
        return self.all_embeds[idx]

class RefDataIterator:
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.iterator = iter(self.dataloader)
        self.device = device
    
    def get_next(self):
        try:
            batch = next(self.iterator)
            return batch.to(self.device) 
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
            return batch.to(self.device)