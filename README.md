# binASR (ASR with Binarized Neural Networks)

Binarized LSTMs, per Ardakani et al. (2018).

To install the requirements, run:

`pip install -r requirements.txt`

## Training
To train a model, run

`python main.py -c path/to/config.yaml -o desired/output/path/ -n exp_name`

An example of the YAML training config, as well as all possible keys can be found in `./config/train.yaml`.
Outputs will be written as follows:

```
output_dir/
  |-- exp_name/
    |-- ckpt/
      |-- e*.pth
    |-- log/
      |-- ...
 
```
Where `ckpt/` contains the model and optimizer checkpoints saved at the step with the lowest CER on the validation set, named `e{step}.pth` and `opt_e{step}.pth`, respectively.

## Testing
To test a model, run 

`python main.py --test -c path/to/config.yaml -o desired/output/path/ -n exp_name --ckpt path/to/checkpoint.pth`

This will run tests on the LibriSpeech `test-clean` set, using a 4-gram language model trained on the LibriSpeech text corpus with `beam_size=500` and `weight=5`.
By design, tests can be run with the same YAML config as training, so you can re-use your training config for testing.

Testing will output a file `<output_dir>/<exp_name>/testing.tsv`. It has 4 columns: 

1. index
2. ground truth
3. beam search transcription
4. greedy transcription

This file can be used to compute the error.

## All Command line arguments

| Name | Abbreviation | Info |
| - | - | - |
| `--test` | None | Runs testing. No Arguments
| `--config` | `-c` | Path to YAML config for training/testing|
| `--output_dir` | `-o` | Directory where tensorboard logs, checkpoints, and testing results will be output. Default is `./` |
| `--name`       | `-n` | Name of experiment. Output will be directed to `<output_dir>/<name>` |
| `--ckpt`       | None           | Path to `.pt[h]` checkpoint file. Only for testing. Required when in test mode. | 

## Data

So far, this repo is only built to run LibriSpeech. Pass the the folder containing the root LibriSpeech folder to `root:` in the `data:` nest in the YAML config.

## Bibliography
```
@article{Ardakani2018,
   author = {Arash Ardakani and Zhengyun Ji and Sean C. Smithson and Brett H. Meyer and Warren J. Gross},
   doi = {10.48550/arxiv.1809.11086},
   journal = {7th International Conference on Learning Representations, ICLR 2019},
   month = {9},
   publisher = {International Conference on Learning Representations, ICLR},
   title = {Learning Recurrent Binary/Ternary Weights},
   url = {https://arxiv.org/abs/1809.11086v2},
   year = {2018},
}
```