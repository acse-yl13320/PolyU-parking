import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, ChebConv, GraphConv, GATv2Conv, ARMAConv, SuperGATConv, TransformerConv
from torch.nn import LSTM, Linear, GRU, RNN

class Graph(torch.nn.Module):
    def __init__(self, num_nodes, t_in, t_out):
        super().__init__()
        self.width = num_nodes
        self.t_in = t_in
        self.t_out = t_out
        
        self.conv = SAGEConv(t_in, t_in)
        # self.conv = GATv2Conv(t_in, t_in)
        # self.conv = GCNConv(t_in, t_in)
        '''
            LSTM init (input_size, output_size, layer)
            LSTM input (batch_size, seq_len, input_size)
            LSTM output (batch_size, seq_len, output_size*2)
        '''
        self.lstm = LSTM(1, 1, num_layers=1, batch_first=True, bidirectional=False)
        # self.gru = GRU(1, 1, num_layers=1, batch_first=True, bidirectional=False)
        # self.rnn = RNN(1, 1, num_layers=1, batch_first=True, bidirectional=True)
        self.lnr = Linear(t_in, t_out)
        
    def forward(self, data):
        '''
            x.shape = (batch_size * width, t_in)
        '''
        # x = self.conv(data.x, data.edge_index)
        
        # x = self.pool(x, edge_index)[0]
        
        # x = F.dropout(x, p=0.2)
        '''
            x.shape = (batch_size * width, t_in)
        '''
        # x = data.x
        # x = x.view(-1, self.t_in, 1)
        '''
            x.shape = (batch * width, seq=t_in, 1)
        '''
        # x, (h, c) = self.lstm(x)
        # x, h = self.gru(x)
        # x, h = self.rnn(x)
        '''
            x.shape = (batch * width, seq=t_in, 1*2)
        '''
        x = data.x
        x = x.reshape(-1, self.t_in)
        x = self.lnr(x)
        # x = self.activate(x)
        '''
            x.shape = (batch_size * width, t_out)
        '''
        return x