import argparse
import pandas as pd
import glob
import os
from argparse import Namespace
from dataclasses import dataclass
from typing import List

import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from models.tts.valle.valle_pgd import VALLEAttackGDBA, VALLEAttackRandom
from utils.util import load_config
from egs.tts.VALLE.const import *

work_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
os.environ['WORK_DIR'] = work_dir 
os.environ['PYTHONPTH'] = work_dir 

print(os.environ['WORK_DIR'])
# exit()

@dataclass
class DataItem:
    uid: str
    audio_file: str
    transcription_file: str

class AttackDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir 
        self.data = self.prepare_data()

    def prepare_data(self):
        wav_files = list(map(
            lambda x: (x.split('.')[0], x),
            glob.glob(os.path.join(self.data_dir, "*.wav"))
        ))
        prompt_files = list(map(
            lambda x: (x.split('.')[0], x),
            glob.glob(os.path.join(self.data_dir, "*.txt"))
        ))

        data = {} 
        ret = []

        for key, val in wav_files:
            data[key] = [val]

        for key, val in prompt_files:
            if key in data:
                data[key].append(val)

        for k,v in data.items():
            if len(v) < 2:
                data.pop(k)
            else:
                uid = os.path.basename(k)
                ret.append(
                    DataItem(
                        uid=uid, 
                        audio_file=v[0],
                        transcription_file=v[1],
                    )
                )

        return ret 
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i].uid, self.data[i].audio_file, self.data[i].transcription_file

class GDBAExecutor:
    def __init__(
            self, 
            attacker: VALLEAttackGDBA, 
            egs_name, 
            data_dir, 
            text, 
            n_inferences=10,
            skip_attack_if_coeff_exists=True
        ):
        self.attacker = attacker 
        self.egs_name = egs_name
        self.data_dir = data_dir 
        self.n_inferences = n_inferences
        self.text = text
        self.skip_attack_if_coeff_exists = skip_attack_if_coeff_exists

    def on_attack_begin(self):
        self.egs_dir = os.path.join(
            os.path.dirname(__file__),
            self.egs_name
        )
        
        os.makedirs(self.egs_dir, exist_ok=True)


        tensorboard_writer = SummaryWriter(os.path.join(self.egs_dir, "tensorboard"))
        
        try:
            attacker.set_tb_writer(tensorboard_writer)
        except:
            pass

        # os.makedirs(os.path.join(os.path.dirname(__file__)))

    def on_attack_item(self, data):
        uid, audio_file, transcription = data
        
        utterance_dir = os.path.join(
            self.egs_dir, 
            uid 
        )

        os.makedirs(utterance_dir, exist_ok=True)

        coeff_path = os.path.join(
            utterance_dir,
            COEFF_NAME
        )

        # Attack
        if not os.path.exists(coeff_path) or not self.skip_attack_if_coeff_exists:
            print(f"{uid}: Optiziming coeff")
            coeff = self.attacker.attack(
                audio_file, transcription
            )

            torch.save(coeff, coeff_path)

        else:
            coeff = torch.load(coeff_path)
            print(f"{COEFF_NAME} found, skipping")

        # Sampling and inference 
        adv_dir = os.path.join(utterance_dir, ADV_NAME)
        inference_dir = os.path.join(utterance_dir, INFERENCE_NAME)

        os.makedirs(adv_dir, exist_ok=True)
        os.makedirs(inference_dir, exist_ok=True)

        # Inference on original audio
        wav = attacker.inference_one_clip(
            text=self.text,
            text_prompt=transcription,
            audio_file=audio_file
        )
        wav_file = os.path.join(
            inference_dir, "orig.wav"
        )

        sf.write(
            wav_file, 
            wav, 
            samplerate=attacker.audio_tokenizer.sample_rate
        )

        print(f"{uid}: Crafting adversarial audio and inference")

        adv_metrics_list = []

        for i in range(self.n_inferences):
            adv_metrics = {}
            
            adv_file = os.path.join(
                adv_dir,
                f"{str(i)}.wav"
            )
            
            adv_metrics['adv_path'] = adv_file
            
            # Sampling from coeff for adversarial audio
            adv_wav, ret_metrics = self.attacker.sample(
                coeff,
                audio_file
            )

            adv_metrics.update(ret_metrics)

            sf.write(
                adv_file, 
                adv_wav, 
                samplerate=attacker.audio_tokenizer.sample_rate
            )

            # Inference on adv sample
            inference_wav = attacker.inference_one_clip(
                text=self.text,
                text_prompt=transcription,
                audio_file=adv_file
            )
            inference_file = os.path.join(
                inference_dir,
                f"{str(i)}.wav"
            )

            sf.write(
                inference_file,
                inference_wav,
                samplerate=attacker.audio_tokenizer.sample_rate
            )

            adv_metrics_list.append(adv_metrics)
        
        df = pd.DataFrame(adv_metrics_list)
        df.to_csv(os.path.join(adv_dir, 'report.csv'), index=0)
        


    def collate_fn(self, batch):
        new_batch = []
        for (uid, audio_file, transcrition_file) in batch:
            with open(transcrition_file, 'r') as f:
                transcription = f.read().strip()
            new_batch.append((uid, audio_file, transcription))

        return new_batch

    def run(self):
        dataset = AttackDataset(self.data_dir)
        dataloader = DataLoader(dataset, batch_size=2, collate_fn=self.collate_fn,)

        self.on_attack_begin() 

        for batch in dataloader:
            for item in batch:
                self.on_attack_item(item)

            # for item in batch:
            #     self.on_attack_item(item)

if __name__ == "__main__":
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

    print("=========== GDBA ATTACK ==============")

    attacker = VALLEAttackGDBA(
        args, cfg,
        lr=1e-3,
        batch_size=6,
        num_iters=10000,
        print_every=10,
        initial_coeff=10
    )

    executor = GDBAExecutor(
        attacker=attacker, 
        egs_name="gdba_attack_steps_10k", 
        data_dir="/home/olli/Desktop/duy/Amphion/egs/tts/VALLE/prompt_examples",
        text="As the sun dipped belown the horizon",
        n_inferences=20,
        skip_attack_if_coeff_exists=True
    )

    executor.run()
    print("=========== RANDOM ATTACK ==============")

    # attacker = VALLEAttackRandom(
    #     args, cfg,
    #     p=0.025, init_coeff=15.
    # )

    # executor = GDBAExecutor(
    #     attacker=attacker, 
    #     egs_name="random_attack", 
    #     data_dir="/home/olli/Desktop/duy/Amphion/egs/tts/VALLE/prompt_examples",
    #     text="As the sun dipped belown the horizon",
    #     n_inferences=20, 
    #     skip_attack_if_coeff_exists=False
    # )

    # executor.run()

