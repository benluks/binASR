from torch import device
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.module import QLSTM, FullyConnected

class BinASRModel(nn.Module):

    def __init__(self, **kwargs):
        
        super().__init__()
        self.input_size = kwargs['input_size']
        self.hidden_size = kwargs['hidden_size']
        self.bias = kwargs['bias']
        self.dropout = kwargs['dropout'] if 'dropout' in kwargs.keys() else 0

        self.output_size = kwargs['output_size']
        self.bidirectional = kwargs['bidirectional']
        
        self.binary = kwargs['binary']
        self.weights_binary = False

        self.device = kwargs['device']

        layers = []
        self.num_layers = kwargs['num_layers']
        
        if 'linear_proj' not in kwargs: kwargs['linear_proj'] = 0
        if kwargs['linear_proj'] != 0:
            self.proj = nn.Sequential(*[FullyConnected(self.input_size, self.hidden_size, bias=self.bias, dropout=self.dropout)])
            for _ in range(kwargs['linear_proj']-1):
                self.proj.append(FullyConnected(self.hidden_size, self.hidden_size, bias=self.bias, dropout=self.dropout))

        else:
            layers.append(
                nn.LSTM(
                    input_size=self.input_size, 
                    hidden_size=self.hidden_size, 
                    batch_first=True,  
                    bias=self.bias, 
                    bidirectional=self.bidirectional,
                    device=self.device
                )
            )

        for i in range(len(layers), self.num_layers):
            if self.binary:
                module = QLSTM
            else:
                module = nn.LSTM
    
            layers.append(
                module(
                    input_size=(1+(self.bidirectional * (i!=0))) * self.hidden_size, 
                    hidden_size=self.hidden_size, 
                    batch_first=True,  
                    bias=self.bias, 
                    bidirectional=self.bidirectional,
                    device=self.device
                )
            )
        self.rnn = nn.Sequential(*layers)
        
        self.fc = nn.Sequential()
        self.fc.add_module('linear', nn.Linear((1+self.bidirectional)*self.hidden_size, self.output_size, bias=self.bias))
        self.fc.add_module('relu', nn.ReLU(inplace=True))
        self.fc.add_module('softmax', nn.LogSoftmax(dim=-1))

    def forward(self, x, lens=None):

        # [B, T, F]
        if hasattr(self, 'proj'):
            x = self.proj(x)
            # [B, T, H]
            if self.binary:
                # normalize mean to 0 so binarization isn't all 1s after ReLU
                x.add_(-x.mean())
        
        if not self.binary and self.training:
            x = pack_padded_sequence(x, lens.cpu().numpy(), batch_first=True)
        
        for rnn_layer in self.rnn:
            x, _ = rnn_layer(x)
        
        if not self.binary and self.training:    
            x, input_lens = pad_packed_sequence(x, batch_first=True)
        y = self.fc(x)
        return y.permute(1, 0, 2)
    

    def reset_binarized_params(self):
        """
        Reset to full-precision params for binary training 
        """
        assert self.weights_binary, "Cannot reset weights that are already full-precision"

        for par in self.parameters():
            if hasattr(par, 'org'):
                par.data = par.org
        self.weights_binary = False


    def save_and_quantize_params(self):
        """
        save full-precision params (weight or bias, not bn)
        and binarize original data
        """
        assert not self.weights_binary, "Cannot binarize weights that are already binarized"

        for mod in self.modules():
            if isinstance(mod, QLSTM) and mod.quant:    
                for name, par in mod.named_parameters():
                    if name[:2] != 'bn':
                        par.org = par.data
                        par.data = mod.binarize(par, name, self.device)
        self.weights_binary = True
        