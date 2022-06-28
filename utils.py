import os.path as osp
import os
import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader
from torch.nn import BCELoss, BCEWithLogitsLoss, MSELoss, L1Loss
import wandb
from tqdm import tqdm
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')
from metrics import generate_score_metrics
from dataset import *
from model import *

def combine_loss(ga_loss, me_loss):
    me_loss /= 50
    return torch.vstack((ga_loss, me_loss))
    # return me_loss
    # return ga_loss

def descale_batch(out, batch_size, num_nodes, t_out, min_v, range_v):
    out = out.view(batch_size, num_nodes, t_out)
    for i in range(batch_size):
        for j in range(t_out):
            out[i, :, j] *= range_v
            out[i, :, j] += min_v
    
    if t_out == 1:
        return out.view(-1)
    else:
        return out.view(-1, t_out)

def test(model, device, loss_fn, test_set: Dataset, test_batch_size=64):
    print('start testing...')
    num_nodes, t_out = test_set[0].y.shape
    
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=True, drop_last=True)
    model.eval()
    test_loss = np.zeros(t_out)
    loss_count = 0
    
    dtype = test_set.dtype
    
    test_score = generate_score_metrics(time_y=test_set.time_y, dtype=dtype)
    
    if dtype == 'combine':
        split_index = test_set[0].split_index
    
    if test_set.scale:
        min_v = test_set.min_v.to(device)
        range_v = test_set.range_v.to(device)
    
    for batch in tqdm(test_loader):
        with torch.no_grad():
            batch = batch.to(device)
            out = model(batch)
            
            if dtype == 'combine':
                ga_out = out[:split_index]
                ga_y = batch.y[:split_index]
                ga_loss = loss_fn[0](ga_out, ga_y)
                
                me_out = out[split_index:]
                me_y = batch.y[split_index:]
                me_loss = loss_fn[1](me_out, me_y)
                
                loss = combine_loss(ga_loss, me_loss)
                
                test_loss += loss.detach().cpu().mean(axis=0).numpy()
                loss_count += 1
                
                if test_set.scale:
                    ga_out = descale_batch(ga_out, -1, split_index, t_out, min_v, range_v)
                    ga_y = descale_batch(ga_y, -1, split_index, t_out, min_v, range_v)
                    
                test_score[0].update(ga_y, ga_out)
                test_score[1].update(me_y, me_out)
            
            else:
                loss = loss_fn(out, batch.y).cpu().mean(axis=0).numpy()
                test_loss += loss
                loss_count += 1
                
                if test_set.scale:
                    out = descale_batch(out, -1, num_nodes, t_out, min_v, range_v)
                    y = descale_batch(batch.y, -1, num_nodes, t_out, min_v, range_v)
                test_score.update(y, out)
    
    test_loss /= loss_count
    loss_dict = {}
    for i, loss in enumerate(test_loss):
        loss_dict['test_loss_t_%d'%(i + 1)] = loss
    loss_dict['test_loss_avg'] = test_loss.mean()

    if dtype == 'combine':
        return loss_dict, test_score[0].score_dict() | test_score[1].score_dict()
    else:
        return loss_dict, test_score.score_dict()
    
def train(train_set: Dataset, test_set: Dataset, name='try', model_path=None, epochs=100, batch_size=32, test_batch_size=64, lr=0.01, resume=False):
    
    current_time = time.strftime('%Y-%m-%d-%H-%M-',time.localtime())
    folder = osp.join('runs', train_set.dtype, current_time + name)
    os.mkdir(folder)
    f = open(osp.join(folder, 'log.txt'), 'w', 1)
    f.write(current_time + '\n')
    f.write('epochs: %d\nbatch_size: %d\nlr: %f\n'%(epochs, batch_size, lr))
    f.write('train_set: %d\ntest_set: %d\n'%(len(train_set), len(test_set)))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    f.write('device: %s\n' % device)
    
    num_nodes, t_in = train_set[0].x.shape
    t_out = train_set[0].y.shape[1]
    f.write('num_nodes: %d\nt_in: %d\nt_out: %d\n'%(num_nodes, t_in, t_out))
    
    if model_path is not None:
        model = torch.load(model_path)
    else:
        model = Graph(num_nodes, t_in, t_out).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    
    period = 10
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i for i in range(epochs) if i % period == period - 1], gamma=0.5)
    
    dtype = train_set.dtype
    if dtype == 'meter':
        loss_fn = BCEWithLogitsLoss(reduction='none')
    elif dtype == 'garage':
        loss_fn = L1Loss(reduction='none')
    elif dtype == 'combine':
        loss_fn = [L1Loss(reduction='none'), BCEWithLogitsLoss(reduction='none')]
        split_index = train_set[0].split_index
        
        
    f.write(str(model))
    f.write('\n')
    f.write(str(optimizer))
    f.write('\n')
    f.write(str(loss_fn))
    f.write('\n')
    
    if resume:
        wandb.init(project=dtype, resume=True)
    else:
        wandb.init(project=dtype, name=name)
    # wandb.watch(model, log='all')
    
    if train_set.scale:
        min_v = train_set.min_v.to(device)
        range_v = train_set.range_v.to(device)
    
    for epoch in range(epochs):
        print('epoch:', epoch)
        print('start training...')
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        
        train_loss = np.zeros(t_out)
        loss_count = 0
        
        train_score = generate_score_metrics(time_y=train_set.time_y, dtype=dtype)
        
        model.train()
        
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            batch = batch.to(device)
            out = model(batch)
            
            if dtype == 'combine':
                
                ga_out = out[:split_index]
                ga_y = batch.y[:split_index]
                ga_loss = loss_fn[0](ga_out, ga_y)
                
                me_out = out[split_index:]
                me_y = batch.y[split_index:]
                me_loss = loss_fn[1](me_out, me_y)
                
                loss = combine_loss(ga_loss, me_loss)
                
                train_loss += loss.detach().cpu().mean(axis=0).numpy()
                loss_count += 1
                
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                
                if train_set.scale:
                    ga_out = descale_batch(ga_out, -1, split_index, t_out, min_v, range_v)
                    ga_y = descale_batch(ga_y, -1, split_index, t_out, min_v, range_v)
                    
                train_score[0].update(ga_y, ga_out)
                train_score[1].update(me_y, me_out)
                
            else:
                loss = loss_fn(out, batch.y)
                train_loss += loss.detach().cpu().mean(axis=0).numpy()
                loss_count += 1
                
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                
                if train_set.scale:
                    out = descale_batch(out, -1, num_nodes, t_out, min_v, range_v)
                    y = descale_batch(batch.y, -1, num_nodes, t_out, min_v, range_v)
                train_score.update(y, out)
        
        # adjust learning rate
        scheduler.step()
            
        train_loss /= loss_count
        loss_dict = {}
        for i, loss in enumerate(train_loss):
            loss_dict['train_loss_t_%d'%(i + 1)] = loss
        loss_dict['train_loss_avg'] = train_loss.mean()
        print(loss_dict)


        if dtype == 'combine':
            train_score_dict = train_score[0].score_dict() | train_score[1].score_dict()
        else:
            train_score_dict = train_score.score_dict()    
        print(train_score_dict)
        
        test_loss_dict, test_score_dict = test(model, device, loss_fn, test_set, test_batch_size=test_batch_size)
        print(test_loss_dict)
        print(test_score_dict)
        
        losses = loss_dict | test_loss_dict
        f.write('epochs: %d\t%s\n'%(epoch, losses))
        f.write('train_score: %s\n'%train_score_dict)
        f.write('test_score: %s\n\n'%test_score_dict)
        
        wandb_dict = losses | train_score_dict | test_score_dict
        wandb.log(wandb_dict)
        
        torch.save(model, osp.join(folder, 'model.pt'))
        
    f.close()

def baseline(dataset: Dataset, method='mean'):
    if dataset.dtype == 'meter':
        loss_fn = BCEWithLogitsLoss()
    elif dataset.dtype == 'garage':
        loss_fn = L1Loss()
    score = generate_score_metrics(time_y=dataset.time_y, dtype=dataset.dtype)
    
    # yesterday
    if method == 'yesterday':
        loss = 0.
        for j, today in enumerate(tqdm(dataset[288:])):
            yesterday = dataset[j - 288]
            loss += loss_fn(yesterday.y[:, 0], today.y[:, 0]).item()
        loss /= (len(dataset) - 288)
        print(print('loss_t_avg:', loss))
        return
        
    # else
    num_nodes, t_out = dataset[0].y.shape
    losses = np.zeros(t_out)
    for data in tqdm(dataset):
        if method == 'mean':
            x = data.x.mean(axis=1)
        elif method == 'last':
            x = data.x[:, -1]
        elif method == '1':
            x = torch.ones_like(data.y[:, 0])
        y = data.y
        new_x = torch.ones_like(y)
        for i in range(y.shape[1]):
            new_x[:, i] = x
            
        if dataset.scale:
            y = descale_batch(y, -1, num_nodes, t_out, dataset.min_v, dataset.range_v)
            new_x = descale_batch(new_x, -1, num_nodes, t_out, dataset.min_v, dataset.range_v)
        score.update(y, new_x)
        for i in range(t_out):
            losses[i] += loss_fn(x, y[:, i]).item()
    
    losses /= len(dataset)
    print(method)
    for i, loss in enumerate(losses):
        print('loss_t_%d:'%(i + 1), loss)
    print('loss_t_avg:', losses.mean())
    print(score.score_dict())
    
    
def test_dataset(model, test_set: Dataset):
    print('start dataset testing...')
    num_nodes, t_out = test_set[0].y.shape
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if type(model) == str:
        assert model in ('last', 'mean')
    else:
        model.eval()
        
    print(model)
    
    loss_matrix = np.zeros((len(test_set), num_nodes))
    
    dtype = test_set.dtype
    
    test_score = generate_score_metrics(time_y=test_set.time_y, dtype=dtype)
    
    if dtype == 'meter':
        loss_fn = BCEWithLogitsLoss(reduction='none')
    elif dtype == 'garage':
        loss_fn = MSELoss(reduction='none')
    elif dtype == 'combine':
        loss_fn = [MSELoss(reduction='none'), BCEWithLogitsLoss(reduction='none')]
        split_index = test_set[0].split_index
    
    if test_set.scale:
        min_v = test_set.min_v.to(device)
        range_v = test_set.range_v.to(device)
    
    for i, data in enumerate(tqdm(test_set)):
        with torch.no_grad():
            data = data.to(device)
            y = data.y
            
            # output
            if type(model) == str:
                if model == 'last':
                    x = data.x[:, -1]
                elif model == 'mean':
                    x = data.x.mean(axis=1)
                out = torch.ones_like(y)
                for j in range(y.shape[1]):
                    out[:, j] = x
            else:
                out = model(data)
            
            # calculate loss
            if dtype == 'combine':
                ga_out = out[:split_index]
                ga_y = y[:split_index]
                ga_loss = loss_fn[0](ga_out, ga_y)
                
                me_out = out[split_index:]
                me_y = y[split_index:]
                me_loss = loss_fn[1](me_out, me_y)
                
                loss = torch.vstack((ga_loss, me_loss))
                
                loss = loss.detach().cpu().numpy()
                
                loss_matrix[i] = loss.mean(axis=1)
                
                if test_set.scale:
                    ga_out = descale_batch(ga_out, -1, split_index, 1, min_v, range_v)
                    ga_y = descale_batch(ga_y, -1, split_index, 1, min_v, range_v)
                    
                test_score[0].update(ga_y, ga_out)
                test_score[1].update(me_y, me_out)
            
            else:
                loss = loss_fn(out, y).cpu().numpy()
                loss_matrix[i] = loss.mean(axis=1)
                
                if test_set.scale:
                    out = descale_batch(out, -1, num_nodes, 1, min_v, range_v)
                    y = descale_batch(y, -1, num_nodes, 1, min_v, range_v)
                test_score.update(y, out)
                
    
    print('avg loss', loss_matrix.mean())
    if dtype == 'combine':
        print('garage loss', loss_matrix[:, :split_index].mean())
        print('meter loss', loss_matrix[:, split_index:].mean())
        
        plt.figure(figsize=(15, 9))
        
        cmap = 'Reds'
        
        plt.subplot(1, 2, 1)
        plt.contour(loss_matrix[:, :split_index], cmap=cmap)
        plt.colorbar()
        
        plt.subplot(1, 2, 2)
        plt.contour(loss_matrix[:, split_index:], cmap=cmap)
        plt.colorbar()
        print('test score ga', test_score[0].score_dict())
        print('test score me', test_score[1].score_dict())
    else:
        plt.figure(figsize=(18, 9))
        plt.contour(loss_matrix)
        plt.colorbar()
        print('test score', test_score.score_dict())
        
    return loss_matrix, test_score
    
    

        
# from torch_geometric.data.collate import collate
# from torch_geometric.data import Batch
# import random     
# class ParkLoader:
#     def __init__(self, dataset: Dataset, batch_size: int):
#         self.dataset = dataset
#         self.batch_size = batch_size
#         index_list = list(range(len(dataset) - batch_size * 2 + 1))
#         random.shuffle(index_list)
#         self.index_list = index_list
#         self.index = 0
        
#     def has_next(self):
#         return self.index < len(self.index_list)
    
#     def len(self):
#         return len(self.index_list)
        
#     def next(self):
#         if not self.has_next():
#             return None
        
#         i = self.index_list[self.index]
#         self.index += 1
        
#         x_batch, _, _ = collate(Batch, [self.dataset[i], self.dataset[i + 1], self.dataset[i + 2]])
#         y_batch, _, _ = collate(Batch, [self.dataset[i + 3], self.dataset[i + 4], self.dataset[i + 5]])
        
#         return x_batch, y_batch

# from torch_geometric.utils import add_self_loops
# from torch_scatter import scatter 
# def min_pool_neighbor_x(data, flow='source_to_target'):
#     r"""Max pools neighboring node features, where each feature in
#     :obj:`data.x` is replaced by the feature value with the maximum value from
#     the central node and its neighbors.
#     """
#     x, edge_index = data.x, data.edge_index

#     edge_index, _ = add_self_loops(edge_index, num_nodes=data.num_nodes)

#     row, col = edge_index
#     row, col = (row, col) if flow == 'source_to_target' else (col, row)

#     data.x = scatter(x[row], col, dim=0, dim_size=data.num_nodes, reduce='min')
#     return data