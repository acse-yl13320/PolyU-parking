from utils import *
from dataset import *


dataset = CombinedDataset(len_x=50)
train_set = dataset[:16000]
test_set = dataset[16000:]

scalar = dataset_scalar(train_set)
train_set.set_scalar(*scalar)
test_set.set_scalar(*scalar)

batch_size = 64
name = 'combine_sage_lstm_lnr_galoss'
n_epochs = 30
lr = 0.001

model_dict = None
# model_dict = torch.load('runs\\garage\\2022-04-17-17-21-lnr_no_act\\model.pth')

train(train_set, test_set, model_dict=model_dict, name=name, epochs=n_epochs, lr=lr, batch_size=batch_size, test_batch_size=batch_size)