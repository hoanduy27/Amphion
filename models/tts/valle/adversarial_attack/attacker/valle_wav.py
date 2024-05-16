import argparse
import glob
import os
import sys
from argparse import Namespace

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from audioseal import AudioSeal
from tqdm import tqdm

from models.tts.valle.adversarial_attack.attacker import VALLEAttackGDBA, VALLEAttackRandom
from utils.util import load_config

work_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
os.environ['WORK_DIR'] = work_dir 
os.environ['PYTHONPTH'] = work_dir 

print(os.environ['WORK_DIR'])

class NoiseGenerator:
    pass

class GaussianGenerator(NoiseGenerator):
    def __init__(self, snr = 1):
        self.snr = snr 

    def generate(self, audio):
        noise = np.random.normal(0, 1, audio.shape)

        audio_power = np.mean(audio ** 2)
        noise_power = np.mean(noise ** 2)

        input_snr = 10 * np.log10(audio_power/noise_power) 

        scale = 10 ** ((input_snr -  self.snr) / 20)

        return audio + noise * scale

class WaveformAttack:
    pass

class GaussianAttack(WaveformAttack):
    def __init__(self, snr=None):
        self.snr = snr 
        if self.snr:
            self.set_snr(snr)

    def set_snr(self, snr):
        self.snr = snr 
        self.noise_gen = GaussianGenerator(snr=snr)
    
    def attack(self, audio_file):
        wav, sr = librosa.load(audio_file, sr=None)

        perturbed_wav = self.noise_gen.generate(wav)

        return perturbed_wav, sr
    
class AudioSealAttack(WaveformAttack):
    def __init__(self, model_path):
        self.model = AudioSeal.load_generator(model_path)

    def attack(self, audio_file):
        wav, sr = torchaudio.load(audio_file)
        wav = wav.unsqueeze(1)
        
        pertured_wav = self.model.get_watermark(wav, sample_rate=sr)

        return pertured_wav.squeeze().detach().cpu().numpy(), sr

def run_attack():
    prompt_folder = "egs/tts/VALLE/prompt_examples"
    wav_files = glob.glob(os.path.join(prompt_folder, '*.wav'))
    snrs = [20, 30, 35, 40, 50]

    # Gaussian attack 
    print("Gaussian attack")
    attacker = GaussianAttack() 
    output_dir = "egs/tts/VALLE/gaussian_attack"

    os.makedirs(output_dir, exist_ok=True)
    for fp in tqdm(wav_files):
        
        utt_id = os.path.splitext(os.path.basename(fp))[0]

        utt_dir = os.path.join(output_dir, utt_id, "adv")
        os.makedirs(utt_dir, exist_ok=True)

        for snr in snrs:
            attacker.set_snr(snr)
            wav, sr = attacker.attack(fp)
            
            filename = os.path.join(
                utt_dir,
                os.path.basename(f"{utt_id}-snr_{snr}.wav")
            )

        sf.write(filename, wav, sr)

    # AudioSeal attack
    print("AudioSeal attack")
    attacker = AudioSealAttack("audioseal_wm_16bits") 
    output_dir = "egs/tts/VALLE/audioseal_attack"
    os.makedirs(output_dir, exist_ok=True)
    for fp in tqdm(wav_files):
        utt_id = os.path.splitext(os.path.basename(fp))[0]

        utt_dir = os.path.join(output_dir, utt_id ,"adv")
        os.makedirs(utt_dir, exist_ok=True)

        wav, sr = attacker.attack(fp)

        filename = os.path.join(
            output_dir,
            os.path.basename(f"{utt_id}.wav")
        )

        sf.write(filename, wav, sr)

def inference():
    # Gaussian attack 
    args = Namespace(
        config = 'egs/tts/VALLE/adv_config.json',
        log_level = "debug",
        acoustics_dir = 'egs/tts/VALLE/valle_librilight_6k/',
        output_dir = 'egs/tts/VALLE/valle_librilight_6k/result',
        mode = "single",
        # text = "As the sun dipped belown the horizon",
        # text_prompt =  "But even the unsuccessful dramatist has his moments",
        # audio_prompt = "egs/tts/VALLE/prompt_examples/7176_92135_000004_000000.wav",
        test_list_file = None,
        dataset = None,
        testing_set = None, 
        speaker_name = None ,
        vocoder_dir = None,
        checkpoint_path = None,
        pitch_control = 1.,
        energy_control = 1.,
        duration_control = 1.,
        top_k=100,
        temperature=1.0,
        continual = False,
        copysyn = False
        # attack args
    )

    cfg = load_config(args.config)
    valle_inference = VALLEAttackGDBA(
        args, cfg,
        lr=1e-3,
        batch_size=8,
        num_iters=1000,
        print_every=10,
        initial_coeff=10
    )


    print("Gaussian attack")
    attacker = GaussianAttack() 
    output_dir = "egs/tts/VALLE/gaussian_attack"

    adv_files = glob.glob(os.path.join(output_dir, '*/adv/*.wav'))

    os.makedirs(output_dir, exist_ok=True)
    for fp in tqdm(adv_files):
        
        utt_id = os.path.splitext(os.path.basename(fp))[0]

        utt_dir = os.path.join(output_dir, utt_id, "adv")
        os.makedirs(utt_dir, exist_ok=True)

        for snr in snrs:
            attacker.set_snr(snr)
            wav, sr = attacker.attack(fp)
            
            filename = os.path.join(
                utt_dir,
                os.path.basename(f"{utt_id}-snr_{snr}.wav")
            )

        sf.write(filename, wav, sr)

    # AudioSeal attack
    print("AudioSeal attack")
    attacker = AudioSealAttack("audioseal_wm_16bits") 
    output_dir = "egs/tts/VALLE/audioseal_attack"
    os.makedirs(output_dir, exist_ok=True)
    for fp in tqdm(wav_files):
        utt_id = os.path.splitext(os.path.basename(fp))[0]

        utt_dir = os.path.join(output_dir, utt_id ,"adv")
        os.makedirs(utt_dir, exist_ok=True)

        wav, sr = attacker.attack(fp)

        filename = os.path.join(
            output_dir,
            os.path.basename(f"{utt_id}.wav")
        )

        sf.write(filename, wav, sr)

if __name__ == "__main__":
    # run_attack()
    inference()