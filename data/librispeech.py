import torch
import torchaudio
from string import ascii_uppercase

labels = ['-', "'", ' '] + list(ascii_uppercase)
label2idx = {label: labels.index(label) for label in labels}

def collate_fn(batch):

    batch_size = len(batch)
    feats, trans = zip(*batch)

    tokenized_trans = [torch.LongTensor([label2idx[letter] for letter in text]) for text in trans]
    
    feat_lens = torch.LongTensor(list(map(len, feats)))
    trans_lens = torch.LongTensor(list(map(len, tokenized_trans)))

    feats_tensor = torch.zeros(batch_size, feat_lens.max(), feats[0].size(-1))
    trans_tensor = torch.zeros(batch_size, trans_lens.max()).long()

    for i in range(batch_size):
        feats_tensor[i, :feat_lens[i], :] = feats[i]
        trans_tensor[i, :trans_lens[i]] = tokenized_trans[i]
    
    feat_lens, perm_idx = feat_lens.sort(0, descending=True)

    feats_tensor = feats_tensor[perm_idx]
    trans_tensor = trans_tensor[perm_idx]
    trans_lens = trans_lens[perm_idx]
    trans = [trans[i] for i in perm_idx]

    return feats_tensor, feat_lens, trans_tensor, trans_lens, trans


class LibriData(torchaudio.datasets.LIBRISPEECH):

  def __init__(self, root, split, feature='fbank', num_mels=23, use_energy=False):
    super().__init__(root=root, url=split)
    self.mels = num_mels
    self.featurizer = getattr(torchaudio.compliance.kaldi, feature.lower())
    self.input_size = self.mels + use_energy

  def __getitem__(self, index):

    waveform, _, trans, _, _, _ = super().__getitem__(index)
    speech_features = self.featurizer(waveform, num_mel_bins=self.mels)

    return speech_features, trans

  def __len__(self):
    return super().__len__()