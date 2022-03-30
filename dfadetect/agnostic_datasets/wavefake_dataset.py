from pathlib import Path

import pandas as pd

from dfadetect.agnostic_datasets.base_dataset import SimpleAudioFakeDataset


WAVEFAKE_KFOLD_SPLIT = {
    0: {
        "train": ['melgan_large', 'waveglow', 'full_band_melgan', 'melgan', 'hifiGAN'],
        "test": ['multi_band_melgan'],
        "val": ['parallel_wavegan'],
        "bonafide_partition": [0.7, 0.15],
        "seed": 42
    },
    1: {
        "train": ['multi_band_melgan', 'melgan_large', 'parallel_wavegan', 'melgan', 'hifiGAN'],
        "test": ['waveglow'],
        "val": ['full_band_melgan'],
        "bonafide_partition": [0.7, 0.15],
        "seed": 43
    },
    2: {
        "train": ['multi_band_melgan', 'melgan_large', 'parallel_wavegan', 'waveglow', 'full_band_melgan'],
        "test": ['melgan'],
        "val": ['hifiGAN'],
        "bonafide_partition": [0.7, 0.15],
        "seed": 44
    }
}


class WaveFakeDataset(SimpleAudioFakeDataset):

    fake_data_path = "generated_audio"
    jsut_real_data_path = "real_audio/jsut_ver1.1/basic5000/wav"
    ljspeech_real_data_path = "real_audio/LJSpeech-1.1/wavs"

    def __init__(self, path, fold_num=0, fold_subset="train", transform=None):
        super().__init__(fold_num, fold_subset, transform)
        self.path = Path(path)

        self.fold_num, self.fold_subset = fold_num, fold_subset
        self.allowed_attacks = WAVEFAKE_KFOLD_SPLIT[fold_num][fold_subset]
        self.bona_partition = WAVEFAKE_KFOLD_SPLIT[fold_num]["bonafide_partition"]
        self.seed = WAVEFAKE_KFOLD_SPLIT[fold_num]["seed"]

        self.samples = pd.concat([self.get_generated_samples(), self.get_real_samples()], ignore_index=True)

    def get_generated_samples(self):
        samples = {
            "user_id": [],
            "sample_name": [],
            "attack_type": [],
            "label": [],
            "path": []
        }

        samples_list = list((self.path / self.fake_data_path).glob("*/*.wav"))
        samples_list = self.filter_samples_by_attack(samples_list)

        for sample in samples_list:
            samples["user_id"].append(None)
            samples["sample_name"].append("_".join(sample.stem.split("_")[:-1]))
            samples["attack_type"].append(self.get_attack_from_path(sample))
            samples["label"].append("spoof")
            samples["path"].append(sample)

        return pd.DataFrame(samples)

    def filter_samples_by_attack(self, samples_list):
        return [s for s in samples_list if self.get_attack_from_path(s) in self.allowed_attacks]

    def get_real_samples(self):
        samples = {
            "user_id": [],
            "sample_name": [],
            "attack_type": [],
            "label": [],
            "path": []
        }

        samples_list = list((self.path / self.jsut_real_data_path).glob("*.wav"))
        samples_list += list((self.path / self.ljspeech_real_data_path).glob("*.wav"))
        samples_list = self.split_real_samples(samples_list)

        for sample in samples_list:
            samples["user_id"].append(None)
            samples["sample_name"].append(sample.stem)
            samples["attack_type"].append("-")
            samples["label"].append("bonafide")
            samples["path"].append(sample)

        return pd.DataFrame(samples)

    @staticmethod
    def get_attack_from_path(path):
        folder_name = path.parents[0].relative_to(path.parents[1])
        return str(folder_name).split("_", maxsplit=1)[-1]


if __name__ == "__main__":
    WAVEFAKE_DATASET_PATH = ""

    real = 0
    fake = 0
    for subset in ['train', 'test', 'val']:
        dataset = WaveFakeDataset(WAVEFAKE_DATASET_PATH, fold_num=2, fold_subset=subset)
        dataset.get_real_samples()
        real += len(dataset)

        print('real', len(dataset))

        dataset = WaveFakeDataset(WAVEFAKE_DATASET_PATH, fold_num=2, fold_subset=subset)
        dataset.get_fake_samples()
        fake += len(dataset)

        print('fake', len(dataset))

    print(real, fake)
