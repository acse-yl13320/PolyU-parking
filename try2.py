from utils import *
from dataset import *


dataset = CombinedDataset(root='2021', len_x=50)
train_set = dataset[:16000]
test_set = dataset[16000:]

scalar = dataset_scalar(train_set)
train_set.set_scalar(*scalar)
test_set.set_scalar(*scalar)

batch_size = 32
name = 'gcn_lstm_lnr_comloss_0.0001'
n_epochs = 30
lr = 0.0001

model_path = None
# model_dict = torch.load('runs\\garage\\2022-04-17-17-21-lnr_no_act\\model.pth')

train(train_set, test_set, model_path=model_path, name=name, epochs=n_epochs, lr=lr, batch_size=batch_size, test_batch_size=batch_size)