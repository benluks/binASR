from math import sqrt
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
import torch.nn.functional as F
from src.util import binarize, qlstm_cell

class QLSTM(nn.LSTM):

    def __init__(self, quant='bin', binarize_inputs=True, bn_inputs=False, **kwargs):

        super().__init__(**kwargs)
        self.init_constant = kwargs['init_constant'] if 'init_constant' in kwargs.keys() else 6.
        self.quant = quant
        self.binarize_inputs = binarize_inputs
        if self.binarize_inputs:
            self.bn_inputs = bn_inputs
        self.device = kwargs['device']

        if self.quant:
            # layer-specific initializations 
            for layer in range(self.num_layers):  
                
                # add batchnorms
                bn_gates = nn.BatchNorm1d(8)
                bn_c = nn.BatchNorm1d(1)
            
                bn_gates.bias.requires_grad_(False)
                bn_c.bias.requires_grad_(False)

                self.add_module(f'bn_l{layer}', bn_gates)
                self.add_module(f'bn_c_l{layer}', bn_c)

                # add scaling factor W0
                l_input_size = self.input_size if layer == 0 else self.hidden_size
                W0_ih = sqrt(self.init_constant / (l_input_size + 4 * self.hidden_size)) / 2
                W0_hh = sqrt(self.init_constant / (self.hidden_size + 4 * self.hidden_size)) / 2

                setattr(self, f'W0_ih_l{layer}', W0_ih)
                setattr(self, f'W0_hh_l{layer}', W0_hh)

                # binarizing inputs
                if self.binarize_inputs:    
                    a0 = sqrt(self.init_constant / l_input_size) / 2
                    setattr(self, f'a0_l{layer}', a0)
                    if self.bn_inputs:
                        bn_a = nn.BatchNorm1d(1)
                        bn_a.bias.requires_grad_(False)
                        self.add_module(f'bn_a_l{layer}', bn_a)
                    
                if self.bidirectional:
                    # add batchnorms
                    bn_gates_reverse = nn.BatchNorm1d(8)
                    bn_c_reverse = nn.BatchNorm1d(1)
                    
                    bn_gates_reverse.bias.requires_grad_(False)
                    bn_c_reverse.bias.requires_grad_(False)
                    
                    self.add_module(f'bn_l{layer}_reverse', bn_gates_reverse)
                    self.add_module(f'bn_c_l{layer}_reverse', bn_c_reverse)


    def _get_layer_params(self, layer, reverse=False):
        """
        Get the appropriate parameters for a given layer during a forward pass
        """
        rev = "_reverse" if reverse else ""
        tail_idx = 2 + len(rev)
        
        layer_params = [p for n, p in self.named_parameters() if n[-tail_idx:] == f"l{layer}{rev}"]
        if not self.bias: layer_params += [0, 0]
        layer_params.append(self.device)
        if self.quant:
            layer_params.append(getattr(self, f'bn_l{layer}{rev}'))
            layer_params.append(getattr(self, f'bn_c_l{layer}{rev}'))
        
        return layer_params


    def binarize(self, par, name, device):
        """
        placeholder to signal which W0 values to pass
        """
        _, place, layer = name.split("_")[:3]
        W0 = getattr(self, f"W0_{place}_{layer}")
        return binarize(par, W0, device=device)


    def forward(self, input, h_0=None):
        if type(input) == nn.utils.rnn.PackedSequence:
            pps = True
            batch_sizes = input.batch_sizes
            T = len(batch_sizes)
            B = batch_sizes[0]
        else:
            pps = False
            T = input.size(0) if not self.batch_first else input.size(1)
            B = input.size(1) if not self.batch_first else input.size(0)
        
        # final hidden states (h and c) for each layer
        h_t = []

        for layer in range(self.num_layers):
            if type(input) == nn.utils.rnn.PackedSequence:
                input = input.data
            layer_params = self._get_layer_params(layer)
            outputs = []

            if self.bidirectional:
                layer_params_reverse = self._get_layer_params(layer, reverse=True)
                outputs_reverse = []
                # hidden states if given h_0 if bidirectional
                if h_0:
                    hidden = (h_0[0][2*layer], h_0[1][2*layer])
                    hidden_reverse = (h_0[0][2*layer+1], h_0[1][2*layer+1])
                else:
                    hidden = 2*(torch.zeros(B, self.hidden_size, device=self.device),)
                    hidden_reverse = 2*(torch.zeros(B, self.hidden_size, device=self.device),)
        
            # init hidden states if not bidirectional
            else:
                if h_0 is not None:
                    hidden = (h_0[0][layer], h_0[1][layer]) if ((self.num_layers > 1) or (h_0[0].dim()==3)) else h_0
                else:
                    hidden = 2*(torch.zeros(B, self.hidden_size, device=self.device),)

            # binarize inputs
            if self.binarize_inputs:
                input = binarize(input, getattr(self, f"a0_l{layer}"), device=self.device)

            if not pps:
                # loop through time steps
                for t in range(T):
                    input_t = input[:, t, :] if self.batch_first else input[t]
                        # print(f"normalized binarized inputs: {input}")
                    hidden = qlstm_cell(input_t, hidden, *layer_params)
                    
                    # maybe binarize hidden activations too
                    
                    outputs.append(hidden[0])

                    if self.bidirectional:
                        input_t_reverse = input[:, -(t+1), :] if self.batch_first else input[-(t+1)]
                        hidden_reverse = qlstm_cell(input_t_reverse, hidden_reverse, *layer_params_reverse)
                        outputs_reverse = [hidden_reverse[0]] + outputs_reverse
            else:
                # input is PackedPaddedSequence
                start = 0
                hidden_t = hidden
                if self.bidirectional:
                    end_reverse = 0
                    hidden_t_reverse = hidden_reverse
                for t in range(len(batch_sizes)):
                    
                    input_t = input[start:start+batch_sizes[t]]

                    hidden_t = hidden_t[0][:batch_sizes[t]], hidden_t[1][:batch_sizes[t]]
                    hidden_t = qlstm_cell(input_t, hidden_t, *layer_params)
                
                    hidden[0][:hidden_t[0].size(0)] = hidden_t[0]
                    hidden[1][:hidden_t[1].size(0)] = hidden_t[1]
                    # tensor [B_t, H] where B_t is the batch size at the current t
                    outputs.append(hidden_t[0])

                    if self.bidirectional:
                        start_reverse = end_reverse-batch_sizes[-(t+1)]
                        input_t_reverse = input[start_reverse:end_reverse] if end_reverse != 0 else input[start_reverse:]
                        
                        hidden_t_reverse = hidden_reverse[0][:batch_sizes[-(t+1)]], hidden_reverse[1][:batch_sizes[-(t+1)]]
                        # print(hidden_t_reverse[0].shape)
                        hidden_t_reverse = qlstm_cell(input_t_reverse, hidden_t_reverse, *layer_params_reverse)
                        
                        hidden_reverse[0][:hidden_t_reverse[0].size(0)] = hidden_t_reverse[0]
                        hidden_reverse[1][:hidden_t_reverse[1].size(0)] = hidden_t_reverse[1]
                        outputs_reverse = [hidden_t_reverse[0]] + outputs_reverse

                        end_reverse -= batch_sizes[-(t+1)]
                        
                    start += batch_sizes[t]
            # all time-steps are done, end T loop
            # -----------------------------------
            h_t.append(hidden)
            if not pps:
                outputs = torch.stack(outputs, 1 if self.batch_first else 0)
            else:
                outputs=torch.cat(outputs)

            # reverse outputs
            if self.bidirectional:
                h_t.append(hidden_reverse)
                # outputs_reverse is shape [B, T, H], we want input to be [B, T, 2*H]
                if not pps:    
                    outputs_reverse = torch.stack(outputs_reverse, 1 if self.batch_first else 0)
                else:
                    outputs_reverse = torch.cat(outputs_reverse)
                outputs = torch.cat((outputs, outputs_reverse), dim=-1)
            
            if pps:
                outputs = PackedSequence(outputs, batch_sizes)
                
            # prev hidden states as following layer's input      
            input = outputs
            
        # h_t is [(h, c), (h, c), ...], we want to separate into lists [[h_l0, h_l1, ...], [c_l0, c_l1, ...]]
        h_t, c_t = list(zip(*h_t))
        h_t, c_t = torch.stack(h_t, 0), torch.stack(c_t, 0)

        return outputs, (h_t, c_t)


class FullyConnected(torch.nn.Module):
    """
    from https://pytorch.org/audio/stable/_modules/torchaudio/models/deepspeech.html
    
    Args:
        n_feature: Number of input features
        n_hidden: Internal hidden unit size.
    """

    def __init__(self, input_size, hidden_size, bias, dropout=0.1, relu_max_clip=20):
        super(FullyConnected, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size, bias=bias)
        self.relu = nn.ReLU()
        self.dropout=dropout
        self.relu_max_clip = relu_max_clip


    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = F.hardtanh(x, 0, self.relu_max_clip)

        if self.dropout:
            x = F.dropout(x, training=self.training)
        return x