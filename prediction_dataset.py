import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import os

class FootballPredictionDataset(Dataset):
    def __init__(self, data_path, split='train', obs_len=8, pred_len=16):
        super().__init__()
        
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.split = split
        
        # load dataset
        if split == 'train':
            file_path = os.path.join(data_path, 'train_clean.p')
        else:
            file_path = os.path.join(data_path, 'test_clean.p')
            
        with open(file_path, 'rb') as f:
            self.data = pickle.load(f)
        

        print("self.data.shape:",self.data.shape)
        
        # Data normalization parameters
        self.mean = np.mean(self.data, axis=(0, 1, 2))
        self.std = np.std(self.data, axis=(0, 1, 2))
        
        # normalized data
        self.data = (self.data - self.mean) / (self.std + 1e-6)
        
        # print(f"Loaded {split} data: {self.data.shape}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        trajectory = self.data[idx]  
        
        obs = trajectory[:self.obs_len]  
        fut = trajectory[self.obs_len:self.obs_len + self.pred_len]  
        
        return {
            'obs': torch.FloatTensor(obs),
            'fut': torch.FloatTensor(fut),
            'full': torch.FloatTensor(trajectory)
        }
    
    def denormalize(self, data):
        """denormalization"""
        return data * self.std + self.mean