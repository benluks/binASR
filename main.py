import torch
from argparse import ArgumentParser
import yaml
from trainer import Trainer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = ArgumentParser(description="Meta Arguments for training/testing binarized ASR")

parser.add_argument('--config', '-c', type=str, help="Path to config file")
args = parser.parse_args()

config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)


if __name__ == '__main__':
    # print(args)
    trainer = Trainer(device=device, **config['hparams'])
    trainer.build(**config)
    
    trainer()



