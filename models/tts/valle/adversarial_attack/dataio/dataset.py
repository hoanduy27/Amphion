import os 
import glob
from dataclasses import dataclass
from torch.utils.data import Dataset

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