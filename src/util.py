
import torch
import torch.nn as nn

def binarize(W, W0, device):
    """
    Binarize normalized weight matrix according to 
    https://arxiv.org/abs/1809.11086
    """
    W_b = torch.clone(W) / W0
    W_b.add_(1).div_(2).clamp_(0, 1)
    mask = torch.rand((W_b.shape), device=device)
    W_b.add_(-mask)
    W_b = W_b.sign()
    return W_b


def qlstm_cell(input, hidden, w_ih, w_hh, b_ih, b_hh, device, 
               bn_gates=nn.Identity(), bn_c=nn.Identity()):

    hx, cx = hidden
    batch_size, hidden_size = hx.shape
    
    # gates: [B, 8*H] => [B, 8, H]
    temp = torch.mm(input, w_ih.t()) + b_ih, torch.mm(hx, w_hh.t()) + b_hh
    temp = torch.cat(temp, dim=1)
    gates = temp.view(batch_size, 8, hidden_size)
    # gates: [B, 8, H] => [B, 2, 4, H] => (sum) => [B, 4, H]
    gates = bn_gates(gates).view(batch_size, 2, 4, hidden_size).sum(1)
    # gates: 4 * ([B, H],)
    ingate, forgetgate, cellgate, outgate = gates.unbind(1)

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    # unsqueeze to allow single batchnorm and resqueeze to get rid of extra dim
    cy = bn_c(((forgetgate * cx) + (ingate * cellgate)).unsqueeze(1)).squeeze(1)
    hy = outgate * torch.tanh(cy)

    return hy, cy


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission, batch_first=False):
        """Given a sequence emission over labels, get the best path
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          List[str]: The resulting transcript
        """
        assert emission.dim() == 3, "decoder takes batched input"
        
        if not batch_first:
            emission = emission.permute(1, 0, 2)
        
        indices = torch.argmax(emission, dim=-1)  # [ [batch_size,] seq_len,]
        
        decoded = []
        for batch in indices:
            batch = torch.unique_consecutive(batch, dim=-1)
            batch = [i for i in batch if i != self.blank]
            decoded.append("".join([self.labels[i] for i in batch]))

        return decoded
