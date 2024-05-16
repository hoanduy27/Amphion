from argparse import Namespace
import argparse
import os 
import soundfile as sf 
import torch 
from models.tts.valle.attacker.valle_pgd import VALLEAttackGDBA, VALLEAttackRandom
from utils.util import load_config
import glob

work_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
os.environ['WORK_DIR'] = work_dir 
os.environ['PYTHONPTH'] = work_dir 

print(os.environ['WORK_DIR'])
# exit()

# parser = 
# args = parser.parse_args()

args = Namespace(
    config = 'egs/tts/VALLE/adv_config.json',
    log_level = "debug",
    acoustics_dir = 'egs/tts/VALLE/valle_librilight_6k/',
    output_dir = 'egs/tts/VALLE/valle_librilight_6k/result',
    mode = "single",
    text = "As the sun dipped belown the horizon",
    text_prompt =  "But even the unsuccessful dramatist has his moments",
    audio_prompt = "egs/tts/VALLE/prompt_examples/7176_92135_000004_000000.wav",
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
# args = Namespace(
#     config = '/home/duy/github/Amphion/egs/tts/VALLE/exp_config.json',
#     log_level = "debug",
#     acoustics_dir = 'egs/tts/VALLE/valle_librilight_6k/',
#     output_dir = 'egs/tts/VALLE/valle_librilight_6k/result',
#     mode = "single",
#     text = "Fish being grilled, scent lingering the air",
#     text_prompt =  "Ten sons sat at meat with him, and I was the youngest.",
#     audio_prompt = "egs/tts/VALLE/prompt_examples/5142_33396_000002_000004.wav",
#     test_list_file = None,
#     dataset = None,
#     testing_set = None, 
#     speaker_name = None ,
#     vocoder_dir = None,
#     checkpoint_path = None,
#     pitch_control = 1.,
#     energy_control = 1.,
#     duration_control = 1.,
#     top_k=100,
#     temperature=1.0,
#     continual = False,
#     copysyn = False
# )

cfg = load_config(args.config)

valle_inference = VALLEAttackGDBA(
    args, cfg,
    lr=1e-3,
    batch_size=8,
    num_iters=1000,
    print_every=10,
    initial_coeff=10
)

# coeff = valle_inference.attack()

# torch.save(coeff, '/home/olli/Desktop/duy/Amphion/egs/tts/VALLE/log_coeffs_gdba.pt')

# exit() 
# coeff = torch.load('/home/olli/Desktop/duy/Amphion/egs/tts/VALLE/log_coeffs_gdba.pt')

snr=50


# Inference
adv_dir = "egs/tts/VALLE/gaussian"
inference_dir = 'egs/tts/VALLE/gaussian_inference'

os.makedirs(adv_dir, exist_ok=True)
os.makedirs(inference_dir, exist_ok=True)

adv_files = glob.glob(os.path.join(adv_dir, f"test.wav"))


for fp in adv_files:
    fname = os.path.basename(fp )
    # Inference origin 
    audio = valle_inference.inference_one_clip(
        text = "As the sun dipped belown the horizon",
        text_prompt =  "But even the unsuccessful dramatist has his moments",
        audio_file = fp,
    )

    sf.write(
        os.path.join(inference_dir, fname), 
        audio, samplerate=valle_inference.audio_tokenizer.sample_rate
    )

# attack random


# attacker = VALLEAttackRandom(
#     args, cfg, p=0.0125
# )
# log_coeffs = attacker.attack("egs/tts/VALLE/prompt_examples/7176_92135_000004_000000.wav")

# dirname = "egs/tts/VALLE/adv_random"
# os.makedirs(dirname, exist_ok=True)
# for i in range(10):

#     sample = attacker.sample(
#         log_coeffs, 
#         audio_file="egs/tts/VALLE/prompt_examples/7176_92135_000004_000000.wav",
#     )

#     filepath = os.path.join(dirname, f'output_{str(i)}.wav')
#     sf.write(filepath, sample, samplerate=attacker.audio_tokenizer.sample_rate)

# inference_dir = 'egs/tts/VALLE/adv_random_inference'
# os.makedirs(inference_dir, exist_ok=True)

# for i in range(10):
#     audio = attacker.inference_one_clip(
#         text = "As the sun dipped belown the horizon",
#         text_prompt =  "But even the unsuccessful dramatist has his moments",
#         audio_file = f"egs/tts/VALLE/adv_random/output_{str(i)}.wav",
#     )

#     sf.write(
#         os.path.join(inference_dir, f'output_{str(i)}.wav'), 
#         audio, samplerate=attacker.audio_tokenizer.sample_rate
#     )