from torch_geometric.data import Dataset, HeteroData, Data
import numpy as np
import os.path as osp
import os
from tqdm import tqdm
import torch


class ParkDataset(Dataset):
    def __init__(self, root, in_memory=False, len_x=10, time_y=[15, 30, 45], transform=None, pre_transform=None, pre_filter=None):
        self.x_idxs = np.arange(len_x)
        self.len_x = len_x
        self.time_y = time_y
        
        self.scale = False
        
        self.y_idxs = np.array(sorted(time_y))
        assert np.all(self.y_idxs % 5 == 0)
        self.y_idxs = self.y_idxs // 5 - 1 + len_x
        self.len_y = len(time_y)
  
        if 'meter' in root:
            self.dtype = 'meter'
        elif 'garage' in root:
            self.dtype = 'garage'
        else:
            self.dtype = 'unknown'
        
        self.in_memory = in_memory
        super().__init__(root, transform, pre_transform, pre_filter)
        if in_memory:
            self.load_in_memory()

    @property
    def processed_file_names(self):
        return os.listdir(self.root)
    
    def set_scalar(self, min_v, max_v):
        self.scale = True
        self.min_v = min_v
        self.range_v = max_v - min_v
    
    def load_in_memory(self):
        if self.in_memory:
            size = len(self.processed_file_names) - 1
            print('loading dataset to memory...')
            xs = []
            for i in tqdm(range(size)):
                x = torch.load(osp.join(self.root, '%d.pt'%i))
                xs.append(x)
            self.x_in_memory = torch.stack(xs, axis=1).float()
            self.link = torch.load(osp.join(self.root, f'link.pt'))

    def len(self):
        return len(self.processed_file_names) - self.y_idxs[-1] - 1

    def get(self, idx):
        
        if self.in_memory:
            X = self.x_in_memory[:, (self.x_idxs + idx)]
            Y = self.x_in_memory[:, (self.y_idxs + idx)]
            link = self.link
        else:
            xs = []
            for i in self.x_idxs:
                x = torch.load(osp.join(self.root, '%d.pt'%(idx + i)))
                xs.append(x)
                # print(i, end=' ')
            X = torch.stack(xs, axis=1).float()
            ys = []
            for j in self.y_idxs:
                y = torch.load(osp.join(self.root, '%d.pt'%(idx + j)))
                ys.append(y)
                # print(j, end=' ')
            Y = torch.stack(ys, axis=1).float()
            link = torch.load(osp.join(self.root, 'link.pt'))
            
        
        if self.scale:
            for i in range(self.len_x):
                X[:, i] = (X[:, i] - self.min_v) / self.range_v
            
            for i in range(self.len_y):
                Y[:, i] = (Y[:, i] - self.min_v) / self.range_v
        
        return Data(x=X, edge_index=link, y=Y)
    
        
class CombinedDataset(Dataset):
    def __init__(self, root='./', meter='meter_dataset', garage='garage_dataset', in_memory=[True, True], len_x=10, time_y=[15, 30, 45], transform=None, pre_transform=None, pre_filter=None):
        self.meter_dataset = ParkDataset(root=osp.join(root, meter), in_memory=in_memory[0], len_x=len_x, time_y=time_y, transform=transform, pre_transform=pre_transform)
        self.garage_dataset = ParkDataset(root=osp.join(root, garage), in_memory=in_memory[1], len_x=len_x, time_y=time_y, transform=transform, pre_transform=pre_transform)
        
        self.link = torch.load(os.path.join(root, 'combine', 'link.pt'))
        
        self.x_idxs = self.meter_dataset.x_idxs
        self.len_x = self.meter_dataset.len_x
        self.time_y = self.meter_dataset.time_y
        
        self.scale = False
        
        self.y_idxs = self.meter_dataset.y_idxs
        self.len_y = self.meter_dataset.len_y
        
        self.dtype = 'combine'
        
        super().__init__(root, transform, pre_transform, pre_filter)
        
    @property
    def processed_file_names(self):
        return self.meter_dataset.processed_file_names

    def len(self):
        return min(len(self.garage_dataset), len(self.meter_dataset))

    def get(self, idx):
        # data = HeteroData()
        # data['meter'] = self.meter_dataset[idx]
        # data['garage'] = self.garage_dataset[idx]
        me_data = self.meter_dataset[idx]
        ga_data = self.garage_dataset[idx]
        
        x = torch.vstack([ga_data.x, me_data.x])
        y = torch.vstack([ga_data.y, me_data.y])
        
        ga_width = ga_data.x.shape[0]
        
        return Data(x=x, y=y, edge_index=self.link, split_index=ga_width)
    
    def set_scalar(self, min_v, max_v):
        self.scale = True
        self.min_v = min_v
        self.range_v = max_v - min_v
        
        self.garage_dataset.set_scalar(min_v, max_v)

def dataset_scalar(dataset):
    if isinstance(dataset, CombinedDataset):
        dataset = dataset.garage_dataset
    
    print('find min max scale...')
    if dataset.in_memory:
        length = len(dataset) + dataset.len_x
        min_v = dataset.x_in_memory[:, :length].min(axis=1)[0]
        max_v = dataset.x_in_memory[:, :length].max(axis=1)[0]
    else:
        min_v = dataset[0].x[:, 0]
        max_v = min_v
        for data in tqdm(dataset[:len(dataset):dataset.len_x]):
            min_v = torch.minimum(data.x.min(axis=1)[0], min_v)
            max_v = torch.maximum(data.x.max(axis=1)[0], max_v)
    
    return min_v, max_v

