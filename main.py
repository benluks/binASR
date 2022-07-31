from argparse import ArgumentParser
from datetime import datetime
import torch
import yaml

from trainer import Trainer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = ArgumentParser(description="Meta Arguments for training/testing binarized ASR")

parser.add_argument('--config', '-c', type=str, help="Path to config file")
parser.add_argument('--name', '-n', type=str, help="Name of experiment", default=datetime.now().strftime("%Y-%m-%d-%H_%M"))
parser.add_argument('--output_dir', '-o', type=str, help="Directory to output logs and checkpoints", default='./')

args = parser.parse_args()

config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    trainer = Trainer(args, **config['hparams'])
    trainer.build(**config)
    
    trainer()