
import os
import copy
import yaml
import torch
from tqdm import tqdm
from functools import partial
from joblib import Parallel, delayed

from core.solver import BaseSolver
from core.asr import ASR
from core.decode import BeamDecoder
from core.data import load_dataset
from core.text import load_text_encoder

from core.audio import create_transform

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Solver(BaseSolver):
    """
    Solver for training
    """

    def __init__(self, config, paras, mode):
        super().__init__(config, paras, mode)

        # # ToDo : support tr/eval on different dataset
        # assert self.config['data']['dataset']['name'] == self.src_config['data']['dataset']['name']
        # self.config['data']['dataset']['path'] = self.src_config['data']['dataset']['path']
        # self.config['data']['dataset']['bucketing'] = False

        # The follow attribute should be identical to training config
        self.config['data']['audio'] = self.src_config['data']['audio']
        self.config['data']['text'] = self.src_config['data']['text']
        self.config['model'] = self.src_config['model']

        # Output file
        self.output_file = str(self.ckpdir) + '_{}_{}.csv'

        # Override batch size for beam decoding
        self.greedy = self.config['decode']['beam_size'] == 1
        if not self.greedy:
            self.config['data']['corpus']['batch_size'] = 1
        else:
            # ToDo : implement greedy
            raise NotImplementedError


        self.audio_transform,self.feat_dim = create_transform(self.config['data']['audio'])
        self.vocab_size = 4337
        self.tokenizer = load_text_encoder(**self.config['data']["text"])

        # _, _, self.feat_dim, self.vocab_size, self.tokenizer, _ = \
        #     load_dataset(self.paras.njobs, self.paras.gpu,
        #                  self.paras.pin_memory, False, **self.config['data'])

    def load_data(self,wav_path):
        ''' 将wav转为fbank特征'''
        # self.dv_set, self.tt_set, self.feat_dim, self.vocab_size, self.tokenizer, msg = \
        #     load_dataset(self.paras.njobs, self.paras.gpu,
        #                  self.paras.pin_memory, False, **self.config['data'])


        # self.verbose(msg)
        file = str(wav_path).split('/')[-1].split('.')[0]
        audio_feat = self.audio_transform(wav_path)
        audio_len = [len(audio_feat)]

        return file,audio_feat,audio_len

    def set_model(self):
        """
        Setup ASR model
        :return:
        """
        # Model
        self.model = ASR(self.feat_dim, self.vocab_size,
                         **self.config['model'])

        # Plug-ins
        if ('emb' in self.config) and (self.config['emb']['enable']) \
                and (self.config['emb']['fuse'] > 0):
            from core.plugin import EmbeddingRegularizer
            self.emb_decoder = EmbeddingRegularizer(
                self.tokenizer, self.model.dec_dim, **self.config['emb'])

        # Load target model in eval mode
        self.load_ckpt()

        # Beam decoder
        self.decoder = BeamDecoder(
            self.model.cpu(), self.emb_decoder, **self.config['decode'])
        self.verbose(self.decoder.create_msg())
        del self.model
        del self.emb_decoder

    def exec(self,wav_dir):
        ''' inference End-to-end ASR system '''

        wav_files = os.listdir(wav_dir)

        for wav_file in wav_files:
            wav_path = os.path.join(wav_dir,wav_file)
            _,audio_feat,audio_len = self.load_data(wav_path)

            results = beam_decode(copy.deepcopy(self.decoder),self.device,audio_feat,audio_len)

            txt_seqs = [self.tokenizer.decode(result) for result in results]
            del results


            print("*"*10+wav_file+"*"*10)

            print("beam search最优路径：")
            print(txt_seqs[0])

            print("beam search搜索路径")

            for b, hyp in enumerate(txt_seqs):
                print(str(b), hyp)
            del txt_seqs
           



def beam_decode(model, device,audio_feat,audio_len):
    # Fetch data : move data/model to device

    feat = audio_feat.unsqueeze(0).to(device)
    print("------------",feat.shape)
    feat_len = torch.tensor(audio_len).to(device)

    model = model.to(device)
    # Decode
    with torch.no_grad():
        hyps = model(feat, feat_len)

    hyp_seqs = [hyp.outIndex for hyp in hyps]
    del hyps
    return hyp_seqs # Note: bs == 1


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Inference LAS.')
    parser.add_argument('--config', type=str, help='Path to experiment config.')
    parser.add_argument('--load', default=None, type=str,
                    help='Load pre-trained model (for training only)', required=False)
    parser.add_argument('--wavpath', default='test/', type=str,help='test wav path.', required=True)

    parser.add_argument('--name', default=None, type=str, help='Name for logging.')
    parser.add_argument('--logdir', default='log/', type=str,
                        help='Logging path.', required=False)
    parser.add_argument('--ckpdir', default='ckpt/', type=str,
                        help='Checkpoint path.', required=False)
    parser.add_argument('--outdir', default='result/', type=str,
                        help='Decode output path.', required=False)
    parser.add_argument('--seed', default=0, type=int,
                        help='Random seed for reproducable results.', required=False)
    parser.add_argument('--cudnn-ctc', action='store_true',
                        help='Switches CTC backend from torch to cudnn')
    parser.add_argument('--njobs', default=32, type=int,
                        help='Number of threads for dataloader/decoding.', required=False)
    parser.add_argument('--cpu', action='store_true', help='Disable GPU training.')
    parser.add_argument('--no-pin', action='store_true',
                        help='Disable pin-memory for dataloader')
    parser.add_argument('--test', action='store_true', help='Test the model.')
    parser.add_argument('--no-msg', action='store_true', help='Hide all messages.')
    parser.add_argument('--lm', action='store_true',
                        help='Option for training RNNLM.')
    # Following features in development.
    parser.add_argument('--amp', action='store_true', help='Option to enable AMP.')
    parser.add_argument('--reserve-gpu', default=0, type=float,
                        help='Option to reserve GPU ram for training.')
    parser.add_argument('--jit', action='store_true',
                        help='Option for enabling jit in pytorch. (feature in development)')
    ###
    paras = parser.parse_args()
    setattr(paras, 'gpu', not paras.cpu)
    setattr(paras, 'pin_memory', not paras.no_pin)
    setattr(paras, 'verbose', not paras.no_msg)


    wav_dir = paras.wavpath
    config = yaml.load(open(paras.config, 'r'), Loader=yaml.FullLoader)
    solver = Solver(config, paras, "test")

    solver.set_model()
    solver.exec(wav_dir)




#python3 inference.py --config ./config/aishell_asr_example_lstm4atthead1_test.yaml   --wavpath test/