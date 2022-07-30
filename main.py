from argparse import ArgumentParser
from datetime import datetime
import torch
import yaml

from trainer import Trainer
from test import TestSolver

parser = ArgumentParser(description="Meta Arguments for training/testing binarized ASR")

parser.add_argument('--train', action='store_true', help="Enable train mode")
parser.add_argument('--test', action='store_true', help="Enable test mode")
parser.add_argument('--config', '-c', type=str, help="Path to config file")
parser.add_argument('--name', '-n', type=str, help="Name of experiment", default=datetime.now().strftime("%Y-%m-%d-%H_%M"))
parser.add_argument('--output_dir', '-o', type=str, help="Directory to output logs and checkpoints", default='./')
parser.add_argument('--ckpt', type=str, help="checkpoint path of model")

args = parser.parse_args()

if (args.train and args.test) or (not (args.train or args.test)):
    parser.error("You must specify at least one of '--train' or '--test' and not both.")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

if args.train: Solver = Trainer
elif args.test: Solver = TestSolver

if args.ckpt:
    config['model']['ckpt'] = args.ckpt
solver = Solver(args, **config)

if __name__ == '__main__':

    solver.build(**config)
    solver()



