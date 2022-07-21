from datetime import datetime
from distutils.dir_util import create_tree
from operator import lt, gt
from pathlib import Path

from jiwer import cer as calculate_cer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.librispeech import LibriData, collate_fn, labels
from model import BinASRModel
from src.util import GreedyCTCDecoder

class Trainer:

  def __init__(self, args, **hparams):
    
    self.hparams = hparams

    self.num_epochs = hparams['num_epochs']
    self.lr = hparams['lr']
    self.batch_size = hparams['batch_size']
    self.binary_training = hparams['binary']
    self.valid_step = hparams['valid_step']
    self.max_step = hparams['max_step']
    
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    self.verbose = hparams['verbose'] if 'verbose' in hparams.keys() else True
    
    # logging and saving
    self.output_dir = Path(args.output_dir)
    self.name = args.name
    self.write_dir = self.output_dir / self.name
    self.write_dir.mkdir(parents=True, exist_ok=True)
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
    # It's important that valid loader be set the `shuffle=False`, because we're logging predictions based 
    # on the sample's index
    self.valid_loader = torch.utils.data.DataLoader(self.valid_set, 
                                                    batch_size=self.batch_size, pin_memory=True, 
                                                    shuffle=False, collate_fn=collate_fn)
    
    
    # decoder for cer
    self.greedy_decoder = GreedyCTCDecoder(labels)
    
    # model
    kwargs['model']['input_size'] = kwargs['data']['num_mels'] + kwargs['data']['use_energy']
    kwargs['model']['output_size'] = len(labels)
    kwargs['model']['binary'] = self.binary_training
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


  def write_progress(self):
    
    final_log = f"[{self.tr_step}/{self.max_step}]:"
    
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
        
    if self.tr_step == self.max_step:
      print(f"""
      Completed training after {self.max_step} training steps with:
        
        Best loss = {round(self.best['loss']['value'], 3)} in step {self.best['loss']['step']}, and
        Best cer = {round(self.best['cer']['value']*100, 3)}% in step {self.best['cer']['step']}.
        """)


  def step(self, batch):

    input, input_lens, target, target_lens, transcripts = batch
    
    # for training, hidden and context aren't necessary
    output = self.model(input.to(self.device), input_lens) 
    loss = self.criterion(output, target.to(self.device), input_lens, target_lens)

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
    cer = calculate_cer(transcripts, predictions)

    return loss, cer, zip(predictions[:5], transcripts[:5])


  def __call__(self):
    self.train()


  def save_model(self):

    # delete old checkpoints
    [f.unlink() for f in self.checkpoint_dir.glob('*')]

    torch.save(self.model.state_dict(), self.checkpoint_dir / f"e{self.tr_step}.pth")
    torch.save(self.optimizer.state_dict(), self.checkpoint_dir / f"opt_e{self.tr_step}.pth")

    print("Saved new best model")


  def log_progress(self, decoded_output):
    for metric in self.latest:
      self.writer.add_scalars(
          metric, 
          {dset: value for dset, value in self.latest[metric].items() if value}, 
          self.tr_step)
    
    # write some text examples
    for idx, (pred, truth) in enumerate(decoded_output):
      if self.tr_step == 0:
        self.writer.add_text(f'true_text_{idx}', truth, self.tr_step)
      self.writer.add_text(f'predicted_text_{idx}', pred, self.tr_step)


  def update_milestone(self):
    
    is_best = False

    for metric in self.latest:
      if self.latest[metric]['valid'] and lt(self.latest[metric]['valid'], self.best[metric]['value']):
        self.best[metric]['value'] = self.latest[metric]['valid']
        self.best[metric]['step'] = self.tr_step
        is_best = True

    if is_best:
      self.save_model()

  def train(self):

    self.best = {
        # computed on validation data
        'loss': {'step': -1, 'value': float('INF')},
        'cer': {'step': -1, 'value': 0}
    }
    self.latest = {
        'loss': {'train': None, 'valid': None},
        'cer': {'train': None, 'valid': None}   
    }

    self.model.train()
    running_loss = 0
    self.tr_step = 0

    pbar = tqdm(total=self.max_step)
    
    while self.tr_step < self.max_step:
      for batch in self.train_loader:
        
        self.optimizer.zero_grad()
        
        loss, _, _, _ = self.step(batch)
        running_loss += loss.item()        

        # backpropagate and optimize
        loss.backward()
        
        # reset to full precision weights here
        if self.binary_training:
          for par in self.model.parameters():
            if hasattr(par, 'org'):
              par.data = par.org

        self.optimizer.step()
        
        if (self.tr_step+1) % self.valid_step == 0:
          print(f"validating step {self.tr_step}...")
          # validation step
          train_loss = running_loss / self.valid_step

          self.model.eval()
          # quantize params since model is in `eval` mode, and thus forward pass
          # will not quantize them
          if self.binary_training:
            self.model.save_and_quantize_params()
          valid_loss, cer, decoded_output = self.validate()

          # update logs checkpoints and milestones
          self.latest['loss']['train'] = train_loss
          self.latest['loss']['valid'] = valid_loss
          self.latest['cer']['valid'] = cer

          self.log_progress(decoded_output)
          self.update_milestone()
          self.write_progress()

          # reset full precision weights so next forward pass doesn't save
          # quantized params as `par.org`, thereby erasing full-precision `org`
          if self.tr_step != self.max_step-1:
            for par in self.model.parameters():
                if hasattr(par, 'org'):
                  par.data = par.org
          
          running_loss = 0
          self.model.train()
        
        self.tr_step += 1
        pbar.update(1)
        if self.tr_step == self.max_step:
          break
    
    pbar.close()
   
    # max step reached; training done; validate one last time
    print(f"validating final step {self.tr_step}...")
    # validation step
    self.model.eval()
    # quantize params since model is in `eval` mode, and thus forward pass
    # will not quantize them
    if self.binary_training:
      self.model.save_and_quantize_params()
    valid_loss, cer, decoded_output = self.validate()

    # update logs checkpoints and milestones
    self.latest['loss']['train'] = train_loss
    self.latest['loss']['valid'] = valid_loss
    self.latest['cer']['valid'] = cer

    self.log_progress(decoded_output)
    self.update_milestone()
    self.write_progress()

          

          
      
      