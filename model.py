import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.module import QLSTM

class BinASRModel(nn.Module):

    def __init__(self, **kwargs):
        
        super().__init__()
        self.input_size = kwargs['input_size']
        self.bias = kwargs['bias']

        self.hidden_size = kwargs['hidden_size']
        self.output_size = kwargs['output_size']
        self.bidirectional = kwargs['bidirectional']
        
        self.binary = kwargs['binary']

        layers = []
        self.num_layers = kwargs['num_layers']

        layers.append(nn.LSTM(self.input_size, self.hidden_size, batch_first=True, 
                                    bias=self.bias, bidirectional=self.bidirectional))

        for _ in range(self.num_layers-1):
            if self.binary:
                layers.append(
                    QLSTM(
                        input_size=(1+self.bidirectional) * self.hidden_size, 
                        hidden_size=self.hidden_size, 
                        batch_first=True,  
                        bias=self.bias, 
                        bidirectional=self.bidirectional
                        )
                    )
            else:
                layers.append(
                    nn.LSTM(
                        (1+self.bidirectional) * self.hidden_size, 
                        self.hidden_size, 
                        batch_first=True, 
                        bias=self.bias, 
                        bidirectional=self.bidirectional
                        )
                    )

        self.rnn = nn.Sequential(*layers)
        
        self.fc = nn.Sequential()
        self.fc.add_module('linear', nn.Linear((1+self.bidirectional)*self.hidden_size, self.output_size, bias=self.bias))
        self.fc.add_module('relu', nn.ReLU(inplace=True))
        self.fc.add_module('softmax', nn.LogSoftmax(dim=-1))

    def forward(self, x, lens=None):

        if not self.binary:
            x = pack_padded_sequence(x, lens.cpu().numpy(), batch_first=True)
        
        for rnn_layer in self.rnn:
            x, _ = rnn_layer(x)
        
        if not self.binary:    
            x, input_lens = pad_packed_sequence(x, batch_first=True)
        y = self.fc(x)
        return y.permute(1, 0, 2)
    
    def save_and_quantize_params(self):
        pass
        