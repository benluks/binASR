from data.librispeech import LibriData, labels
from model import BinASRModel

from jiwer import wer
from pathlib import Path
import torch
from torchaudio.models.decoder import ctc_decoder, download_pretrained_files
from tqdm import tqdm
from src.util import GreedyCTCDecoder


class Tester:
    def __init__(self, args, **kwargs):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.use_lm = kwargs['lm'] if 'lm' in kwargs.keys() else True
        
        if self.use_lm:
            self.tokens = [label.lower() for label in labels]
            self.tokens[self.tokens.index(' ')] = "|"
            self.LM_WEIGHT = 5
            self.WORD_SCORE = 0
            self.BEAM_SIZE = kwargs['beam_size'] if 'beam_size' in kwargs.keys() else 500
            
            self.output_dir = Path(args.output_dir)
            self.name = args.name
            self.write_dir = self.output_dir / self.name
            self.write_dir.mkdir(parents=True, exist_ok=True)


    def build(self, **kwargs):
        """
        Dedicated function to execute more storage- and resource-intensive 
        initializations
        """

        # data
        if 'train_split' in kwargs['data'].keys():
            kwargs['data'].pop('train_split')
        if 'eval_split' in kwargs['data'].keys():
            kwargs['data'].pop('eval_split')

        kwargs['data']['split'] = 'test-clean'
        self.test_set = LibriData(**kwargs['data'])
        # self.test_loader = torch.utils.data.DataLoader(self.train_set, 
        #                                                 batch_size=self.batch_size, pin_memory=True, 
        #                                                 shuffle=True, collate_fn=collate_fn)
        
        # model
        kwargs['model']['input_size'] = kwargs['data']['num_mels'] + kwargs['data']['use_energy']
        kwargs['model']['output_size'] = len(labels)

        kwargs['model']['binary'] = kwargs['hparams']['binary']
        kwargs['model']['device'] = self.device
        self.model = BinASRModel(**kwargs['model']).to(self.device)
        self.model.load_state_dict(torch.load(kwargs['model']['ckpt'], map_location=self.device))
        self.model.eval()

        lm_files = download_pretrained_files('librispeech-4-gram')
        self.beam_search_decoder = ctc_decoder(
            lexicon = lm_files.lexicon,
            tokens = self.tokens,
            lm = lm_files.lm,
            beam_size = self.BEAM_SIZE,
            lm_weight = self.LM_WEIGHT,
            word_score = self.WORD_SCORE
        )
        self.greedy_decoder = GreedyCTCDecoder(labels)

        print(f"""
        Model: {self.model}
        __________________
        
        Decoder
        
        
        __________________
        
        Test data length: {len(self.test_set)} 
        __________________
        """)

        print('Successfully built!')
        

    def __call__(self):
        self.test()
        


    def beam_search(self, emission):
        """
        Perform beam search decoding on single-element batch
        """
        beam_search_result = self.beam_search_decoder(emission.permute(1, 0, 2))
        beam_search_transcript = " ".join(beam_search_result[0][0].words).strip()

        return beam_search_transcript


    def test(self):

        outf = self.write_dir/"testing.csv"

        start_idx = 0
        if outf.is_file():
            start_idx = sum(1 for _ in open(outf))

        with open(self.write_dir/"testing.csv", 'a') as f:
            
            for i in tqdm(range(start_idx, len(self.test_set))):
                feats, trans = self.test_set[i]
                emission = self.model(feats.unsqueeze(0).to(self.device))

                beam_search_transcript = self.beam_search(emission.to(torch.device('cpu'))).upper()
                greedy_transcript = self.greedy_decoder(emission)[0]

                f.write(f"{i}\t{trans}\t{beam_search_transcript}\t{greedy_transcript}\n")
                
        


