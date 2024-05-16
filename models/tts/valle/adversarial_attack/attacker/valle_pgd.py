import os
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import argparse
import soundfile as sf

from text.g2p_module import G2PModule
from utils.tokenizer import AudioTokenizer, tokenize_audio
from models.tts.valle.valle import AdvVALLE
from models.tts.base.tts_inferece import TTSInference
from models.tts.valle.valle_dataset import VALLETestDataset, VALLETestCollator
from processors.phone_extractor import phoneExtractor
from text.text_token_collation import phoneIDCollation

from encodec import EncodecModel
from encodec.utils import convert_audio

from models.tts.valle.valle_inference import VALLEInference
from copy import deepcopy

def log_perplexity(logits, coeffs):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_coeffs = coeffs[:, 1:, :].contiguous()
    shift_logits = shift_logits[:, :, :shift_coeffs.size(2)]
    return -(shift_coeffs * F.log_softmax(shift_logits, dim=-1)).sum(-1).mean()


class VALLEAttackGDBA(VALLEInference):
    def __init__(self, args=None, cfg=None,
                 initial_coeff=15,
                 lr=1e-2,
                 batch_size=5,
                 num_iters=1000,
                 print_every=10
    ):
        super(VALLEAttackGDBA, self).__init__(args, cfg)
        self.initial_coeff = initial_coeff
        self.lr = lr 
        self.num_iters = num_iters
        self.batch_size = batch_size
        self.supported_mode = ["default", "targeted"]
        self.args = args 
        self.cfg = cfg
        self.print_every = print_every

    def _build_model(self):
        model = AdvVALLE(self.cfg.model)
        return model

    def tokenize_audio(self, audio_path: str):
        """
        Tokenize the audio waveform using the given AudioTokenizer.

        Args:
            tokenizer: An instance of AudioTokenizer.
            audio_path: Path to the audio file.

        Returns:
            A tensor of encoded frames from the audio.

        Raises:
            FileNotFoundError: If the audio file is not found.
            RuntimeError: If there's an error processing the audio data.
        """
        # try:
        # Load and preprocess the audio waveform
        wav, sr = torchaudio.load(audio_path)
        wav = convert_audio(wav, sr, self.audio_tokenizer.sample_rate, self.audio_tokenizer.channels)
        wav = wav.unsqueeze(0)

        # Extract discrete codes from EnCodec
        # with torch.no_grad():
        encoded_frames = self.audio_tokenizer.encode(wav)
        return wav, encoded_frames

        # except FileNotFoundError:
        #     raise FileNotFoundError(f"Audio file not found at {audio_path}")
        # except Exception as e:
        #     raise RuntimeError(f"Error processing audio data: {e}")
    

    def attack(self):
        # get phone symbol filetext = self.args.text
        text = self.args.text
        text_prompt = self.args.text_prompt
        audio_file = self.args.audio_prompt
        phone_symbol_file = None


        if self.cfg.preprocess.phone_extractor != "lexicon":
            phone_symbol_file = os.path.join(
                self.exp_dir, self.cfg.preprocess.symbols_dict
            )
            assert os.path.exists(phone_symbol_file)
        # convert text to phone sequence
        phone_extractor = phoneExtractor(self.cfg)
        # convert phone sequence to phone id sequence
        phon_id_collator = phoneIDCollation(
            self.cfg, symbols_dict_file=phone_symbol_file
        )

        # text = f"{text_prompt}".strip()
        phone_seq = phone_extractor.extract_phone(text_prompt.strip())  # phone_seq: list
        phone_id_seq = phon_id_collator.get_phone_id_sequence(self.cfg, phone_seq)
        phone_id_seq_len = torch.IntTensor([len(phone_id_seq)]).to(self.device)

        # convert phone sequence to phone id sequence
        phone_id_seq = np.array([phone_id_seq])
        phone_id_seq = torch.from_numpy(phone_id_seq).to(self.device)

        # tokenizer audio
        wav, encoded_frames = self.tokenize_audio(audio_file)
        
        # (1, T, 8)
        audio_prompt_token = encoded_frames[0][0].transpose(2, 1).to(self.device)

        audio_prompt_token_len = torch.IntTensor(
            [audio_prompt_token.shape[1]]
        ).to(self.device)
        # Compute prediction
        # encoded_frames, logits = self.model.inference(
        #     phone_id_seq,
        #     phone_id_seq_len,
        #     audio_prompt_token,
        #     enroll_x_lens=prompt_phone_id_seq_len,
        #     top_k=self.args.top_k,
        #     temperature=self.args.temperature,
        # )

        

        # adv_loss = torch.zeros()
        
        # Get embeddings
        with torch.no_grad():
            # (V, D)
            embeddings = self.model.ar_audio_embedding(
                torch.arange(
                    0, 
                    self.model.cfg.audio_token_num
                ).to(self.device)
            )
        
        T = audio_prompt_token.shape[1]
        V = embeddings.size(0)

        loss, _, info = self.model(
            phone_id_seq,
            phone_id_seq_len,
            audio_prompt_token,
            audio_prompt_token_len,
            reduction="mean"
        )   

        print(f"At init: {loss=}, {info['ar_loss']=}")

        # forbidden = np.zeros(
        #     T
        # ).astype('bool')

        # # Avoid sampling to EOS
        # forbidden[self.model.cfg.audio_token_num] = True
        # if self.model.cfg.prepend_bos:  
        #     # Avoid sampling to SOS
        #     forbidden[self.model.cfg.audio_token_num + 1] = True

        # forbidden_indices = np.arange(0, T)[forbidden]
        # forbidden_indices = torch.from_numpy(forbidden_indices).to(self.device)

        # init coeff
        with torch.no_grad():
            log_coeffs = torch.zeros(T, V).to(self.device)
            indices = torch.arange(log_coeffs.size(0)).long()
            log_coeffs[indices, audio_prompt_token[0,:,0]] = self.initial_coeff
            log_coeffs = log_coeffs
            log_coeffs.requires_grad = True

        optimizer = torch.optim.Adam([log_coeffs], lr=self.lr)
        for i in range(self.num_iters):
            optimizer.zero_grad()

            # (B, T, V)
            coeffs = F.gumbel_softmax(
                log_coeffs.unsqueeze(0).repeat(self.batch_size, 1, 1),
                hard=False
            )

            # (B, T, D)
            inputs_embeds = (coeffs @ embeddings[None, :, :])

            # adv
            _, _, info = self.model(
                phone_id_seq.repeat(self.batch_size, 1),
                phone_id_seq_len.repeat(self.batch_size),
                audio_prompt_token.repeat(self.batch_size, 1, 1),
                audio_prompt_token_len.repeat(self.batch_size),
                inputs_embeds, 
                reduction="mean"
            )

            adv_loss = -info['ar_loss']

            # perplexity
            # (B, V, T) -> (B, T, V)
            logits = info['ar_logits'].permute(0, 2, 1)
            perplexity_loss = log_perplexity(logits, coeffs)

            # similariy
            sim_loss = F.cross_entropy(log_coeffs, audio_prompt_token[0,:,0], reduction="mean")

            total_loss = adv_loss + perplexity_loss + sim_loss
            total_loss.backward() 

            entropy = torch.sum(-F.log_softmax(log_coeffs, dim=1) * F.softmax(log_coeffs, dim=1))
            if i % self.print_every == 0:
                print(f'Iteration {i+1}: {total_loss.item()=}, {adv_loss.item()=}, {perplexity_loss.item()=}, {sim_loss.item()=}, {entropy.item()=}')
                print(log_coeffs)

            # Step
            # log_coeffs.grad.index_fill_(0, forbidden_indices, 0)
            optimizer.step()
        

        # torch.save(log_coeffs, '/home/olli/Desktop/duy/Amphion/egs/tts/VALLE/log_coeffs.pt')
            
        return log_coeffs

    def sample(self, log_coeffs, audio_file):
        wav, encoded_frames = self.tokenize_audio(audio_file)

        ar_codes = deepcopy(encoded_frames[0][0][0][0])

        # (1, T, 8)
        adv_ids = F.gumbel_softmax(log_coeffs, hard=True).argmax(1)

        encoded_frames[0][0][0][0] = adv_ids

        print((ar_codes != adv_ids).sum())
        print((ar_codes != adv_ids).sum() * 100 / len(adv_ids))

        wav = self.audio_tokenizer.decode(encoded_frames)

            # os.makedirs(output_dir, exist_ok=True )

            # filepath = os.path.join(output_dir, f'output_{str(i)}.wav')
            
            # sf.write(filepath, wav.squeeze().detach().cpu().numpy(), samplerate=self.audio_tokenizer.sample_rate)

        # self.audio_tokenizer.decode

        # audio_prompt_token[0] = adv_ids

        return wav.squeeze().detach().cpu().numpy()
    
class VALLEAttackRandom(VALLEInference):
    def __init__(self, args=None, cfg=None,
                    p=0.025
    ):
        super(VALLEAttackRandom, self).__init__(args, cfg)
        self.p = p

    def _build_model(self):
        model = AdvVALLE(self.cfg.model)
        return model

    def tokenize_audio(self, audio_path: str):
        """
        Tokenize the audio waveform using the given AudioTokenizer.

        Args:
            tokenizer: An instance of AudioTokenizer.
            audio_path: Path to the audio file.

        Returns:
            A tensor of encoded frames from the audio.

        Raises:
            FileNotFoundError: If the audio file is not found.
            RuntimeError: If there's an error processing the audio data.
        """
        # try:
        # Load and preprocess the audio waveform
        wav, sr = torchaudio.load(audio_path)
        wav = convert_audio(wav, sr, self.audio_tokenizer.sample_rate, self.audio_tokenizer.channels)
        wav = wav.unsqueeze(0)

        # Extract discrete codes from EnCodec
        # with torch.no_grad():
        encoded_frames = self.audio_tokenizer.encode(wav)
        return wav, encoded_frames

        # except FileNotFoundError:
        #     raise FileNotFoundError(f"Audio file not found at {audio_path}")
        # except Exception as e:
        #     raise RuntimeError(f"Error processing audio data: {e}")
    

    def attack(self, audio_file):
        wav, encoded_frames = self.tokenize_audio(audio_file)

        # (1, T, 8)
        audio_prompt_token = encoded_frames[0][0].transpose(2, 1).to(self.device)

        T = len(encoded_frames[0][0][0][0])
        V = self.model.cfg.audio_token_num

        with torch.no_grad():
            log_coeffs = torch.zeros(T, V).to(self.device)
            # Choose indices for corruption with prob p
            indices = torch.arange(T)
            corrupted_indices = indices[torch.rand(T) <= self.p] 
            
            log_coeffs[indices, audio_prompt_token[0,:,0]] = 1
            
            log_coeffs[indices[corrupted_indices]] = 1 - log_coeffs[indices[corrupted_indices]]

            log_coeffs[log_coeffs == 0] = -1e6

        return log_coeffs


    def sample(self, log_coeffs, audio_file):
        wav, encoded_frames = self.tokenize_audio(audio_file)

        ar_codes = deepcopy(encoded_frames[0][0][0][0])

        # (1, T, 8)

        # for i in range(n):
        adv_ids = F.gumbel_softmax(log_coeffs, hard=True).argmax(1)

        encoded_frames[0][0][0][0] = adv_ids

        print((ar_codes != adv_ids).sum())
        print((ar_codes != adv_ids).sum() * 100 / len(adv_ids))

        wav = self.audio_tokenizer.decode(encoded_frames)

        return wav.squeeze().detach().cpu().numpy()

            # os.makedirs(output_dir, exist_ok=True )

            # filepath = os.path.join(output_dir, f'output_{str(i)}.wav')
            
            # sf.write(filepath, wav.squeeze().detach().cpu().numpy(), samplerate=self.audio_tokenizer.sample_rate)

        # self.audio_tokenizer.decode

        # audio_prompt_token[0] = adv_ids

        

        