from datetime import datetime
from operator import lt, gt
from pathlib import Path

from jiwer import wer, cer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.librispeech import LibriData, collate_fn, labels
from model import BinASRModel
from src.util import GreedyCTCDecoder

class Trainer:

  def __init__(self, **hparams):
    
    self.hparams = hparams
    self.device = hparams['device']

    self.num_epochs = hparams['num_epochs']
    self.lr = hparams['lr']
    self.batch_size = hparams['batch_size']
    
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    self.verbose = hparams['verbose'] if 'verbose' in hparams.keys() else True
    
    # logging and saving
    self.output_dir = Path(hparams['output_dir']) if 'output_dir' in hparams else Path('.')
    self.name = hparams['name'] if 'name' in hparams else datetime.now().strftime("%Y-%m-%d-%H_%M")
    self.write_dir = self.output_dir / self.name
    self.writer = SummaryWriter(log_dir=self.write_dir / 'log/')
    self.checkpoint_dir = self.write_dir / 'ckpt'


  def build(self, **kwargs):
    """
    Dedicated function to execute more storage- and resource-intensive 
    initializations
    """

    # data
    train_split = kwargs['data'].pop('train_split')
    eval_split = kwargs['data'].pop('eval_split')
    
    self.train_set = LibriData(split=train_split, **kwargs['data'])
    self.valid_set = LibriData(split=eval_split, **kwargs['data'])
    
    self.train_loader = torch.utils.data.DataLoader(self.train_set, 
                                                    batch_size=self.batch_size, pin_memory=True, 
                                                    shuffle=True, collate_fn=collate_fn)
    self.valid_loader = torch.utils.data.DataLoader(self.valid_set, 
                                                    batch_size=self.batch_size, pin_memory=True, 
                                                    shuffle=True, collate_fn=collate_fn)
    
    
    # decoder for cer
    self.greedy_decoder = GreedyCTCDecoder(labels)
    
    # model
    kwargs['model']['input_size'] = kwargs['data']['num_mels'] + kwargs['data']['use_energy']
    kwargs['model']['output_size'] = len(labels)
    self.model = BinASRModel(**kwargs['model']).to(self.device)

    # optimization
    self.optimizer = getattr(torch.optim, kwargs['hparams']['optimizer'])(self.model.parameters(), lr=self.lr)
    self.criterion = nn.CTCLoss(blank=0)

    self.checkpoint_dir.mkdir(exist_ok=True)
    if self.verbose:
      print(f"""
      Model: {self.model}
      __________________
      
      Training data length: {len(self.train_set)} 
      Validation data length: {len(self.valid_set)}
      __________________

      Optimizer: {self.optimizer}
      __________________
      """)

    print(f'Successfully built! logs and checkpoint models can be found at {self.write_dir.resolve()}')


  def write_progress(self, epoch):
    
    final_log = f"[{epoch}/{self.num_epochs}]:"
    
    for metric in self.latest:
      for dset in self.latest[metric]:
        value = self.latest[metric][dset]
        if value:
        # exclude NoneType
          value = f"{round(value * 100, 3)}%" if metric == 'cer' else round(value, 3)
          final_log += f" {f'{dset} {metric}'.capitalize()}: {value}"
          final_log += " |"
    final_log += "|"

    print(final_log)
        
    if epoch == self.num_epochs:
      print(f"""
      Completed training after {self.num_epochs} epochs with:
        
        Best loss = {round(self.best['loss']['value'], 3)} in epoch {self.best['loss']['epoch']}, and
        Best cer = {round(self.best['cer']['value']*100, 3)}% in epoch {self.best['cer']['epoch']}.
        """)


  def step(self, batch):

    input, input_lens, target, target_lens, transcripts = batch
    
    # for training, hidden and context aren't necessary
    output = self.model(input, input_lens) 
    loss = self.criterion(output, target, input_lens, target_lens)

    return loss, output, target, transcripts


  def validate(self):
    
    running_loss = 0
    predictions = []
    transcripts = []
    
    with torch.no_grad():
      for batch in tqdm(self.valid_loader, position=0):

        loss, output, _, trans = self.step(batch)
        running_loss += loss.item()
        
        # compute cer
        predictions += self.greedy_decoder(output)
        transcripts += trans

    loss = running_loss / len(self.valid_loader)
    error = cer(transcripts, predictions)

    return loss, error


  def __call__(self):
    self.train()


  def save_model(self, epoch):

    # delete old checkpoints
    [f.unlink() for f in self.checkpoint_dir.glob('*')]

    torch.save(self.model.state_dict(), self.checkpoint_dir / f"e{epoch}.pth")
    torch.save(self.optimizer.state_dict(), self.checkpoint_dir / f"opt_e{epoch}.pth")

    print("Saved new best model")


  def log_progress(self, epoch):
    for metric in self.latest:
      self.writer.add_scalars(
          metric, 
          {dset: value for dset, value in self.latest[metric].items() if value}, 
          epoch)


  def update_milestone(self, epoch):
    
    is_best = False

    for metric in self.latest:
      if self.latest[metric]['valid'] and lt(self.latest[metric]['valid'], self.best[metric]['value']):
        self.best[metric]['value'] = self.latest[metric]['valid']
        self.best[metric]['epoch'] = epoch
        is_best = True

    if is_best:
      self.save_model(epoch)

  def train(self):

    self.best = {
        # computed on validation data
        'loss': {'epoch': -1, 'value': float('INF')},
        'cer': {'epoch': -1, 'value': 0}
    }
    self.latest = {
        'loss': {'train': None, 'valid': None},
        'cer': {'train': None, 'valid': None}   
    }

    for epoch in range(self.num_epochs):
      
      running_loss = 0
      
      self.model.train()
      for batch in tqdm(self.train_loader, position=0):
        
        self.optimizer.zero_grad()
        
        loss, _, _, _ = self.step(batch)
        running_loss += loss.item()        

        # backpropagate and optimize
        loss.backward()
        # reset to full precision weights here
        for par in self.model.parameters():
          if hasattr(par, 'org'):
            par.data = par.org

        self.optimizer.step()
        
      train_loss = running_loss / len(self.train_loader)

      self.model.eval()
      # quantize params since model is in `eval` mode, and thus forward pass
      # will not quantize them
      self.model.save_and_quantize_params()
      valid_loss, error = self.validate()

      # reset full precision weights so next forward pass doesn't save
      # quantized params as `par.org`, thereby erasing full-precision `org`
      if epoch != self.num_epochs-1:
        for par in self.model.parameters():
            if hasattr(par, 'org'):
              par.data = par.org
      
      self.latest['loss']['train'] = train_loss
      self.latest['loss']['valid'] = valid_loss
      self.latest['cer']['valid'] = error

      self.log_progress(epoch+1)
      self.update_milestone(epoch+1)
      self.write_progress(epoch+1)