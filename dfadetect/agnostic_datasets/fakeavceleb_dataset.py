from pathlib import Path

import pandas as pd

from dfadetect.agnostic_datasets.base_dataset import SimpleAudioFakeDataset


FAKEAVCELEB_KFOLD_SPLIT = {
    0: {
        "train": ['rtvc', 'faceswap-wav2lip'],
        "test": ['fsgan-wav2lip'],
        "val": ['wav2lip'],
        "bonafide_partition": [0.7, 0.15],
        "seed": 42
    },
    1: {
        "train": ['fsgan-wav2lip', 'wav2lip'],
        "test": ['rtvc'],
        "val": ['faceswap-wav2lip'],
        "bonafide_partition": [0.7, 0.15],
        "seed": 43
    },
    2: {
        "train": ['faceswap-wav2lip', 'fsgan-wav2lip'],
        "test": ['wav2lip'],
        "val": ['rtvc'],
        "bonafide_partition": [0.7, 0.15],
        "seed": 44
    }
}


class FakeAVCelebDataset(SimpleAudioFakeDataset):

    audio_folder = "FakeAVCeleb-audio"
    audio_extension = ".mp3"
    metadata_file = Path(audio_folder) / "meta_data.csv"
    subsets = ("train", "dev", "eval")

    def __init__(self, path, fold_num=0, fold_subset="train", transform=None):
        super().__init__(fold_num, fold_subset, transform)
        self.path = path

        self.fold_num, self.fold_subset = fold_num, fold_subset
        self.allowed_attacks = FAKEAVCELEB_KFOLD_SPLIT[fold_num][fold_subset]
        self.bona_partition = FAKEAVCELEB_KFOLD_SPLIT[fold_num]["bonafide_partition"]
        self.seed = FAKEAVCELEB_KFOLD_SPLIT[fold_num]["seed"]

        self.metadata = self.get_metadata()

        self.samples = pd.concat([self.get_fake_samples(), self.get_real_samples()], ignore_index=True)

    def get_metadata(self):
        md = pd.read_csv(Path(self.path) / self.metadata_file)
        md["audio_type"] = md["type"].apply(lambda x: x.split("-")[-1])
        return md

    def get_fake_samples(self):
        samples = {
            "user_id": [],
            "sample_name": [],
            "attack_type": [],
            "label": [],
            "path": []
        }

        for attack_name in self.allowed_attacks:
            fake_samples = self.metadata[
                (self.metadata["method"] == attack_name) & (self.metadata["audio_type"] == "FakeAudio")
            ]

            for index, sample in fake_samples.iterrows():
                samples["user_id"].append(sample["source"])
                samples["sample_name"].append(Path(sample["filename"]).stem)
                samples["attack_type"].append(sample["method"])
                samples["label"].append("spoof")
                samples["path"].append(self.get_file_path(sample))

        return pd.DataFrame(samples)

    def get_real_samples(self):
        samples = {
            "user_id": [],
            "sample_name": [],
            "attack_type": [],
            "label": [],
            "path": []
        }

        samples_list = self.metadata[
            (self.metadata["method"] == "real") & (self.metadata["audio_type"] == "RealAudio")
        ]

        samples_list = self.split_real_samples(samples_list)

        for index, sample in samples_list.iterrows():
            samples["user_id"].append(sample["source"])
            samples["sample_name"].append(Path(sample["filename"]).stem)
            samples["attack_type"].append("-")
            samples["label"].append("bonafide")
            samples["path"].append(self.get_file_path(sample))

        return pd.DataFrame(samples)

    def get_file_path(self, sample):
        path = "/".join([self.audio_folder, *sample["path"].split("/")[1:]])
        return Path(self.path) / path / Path(sample["filename"]).with_suffix(self.audio_extension)


if __name__ == "__main__":
    FAKEAVCELEB_DATASET_PATH = ""

    real = 0
    fake = 0
    for subset in ['train', 'test', 'val']:
        dataset = FakeAVCelebDataset(FAKEAVCELEB_DATASET_PATH, fold_num=2, fold_subset=subset)
        dataset.get_real_samples()
        real += len(dataset)

        print('real', len(dataset))

        dataset = FakeAVCelebDataset(FAKEAVCELEB_DATASET_PATH, fold_num=2, fold_subset=subset)
        dataset.get_fake_samples()
        fake += len(dataset)

        print('fake', len(dataset))

    print(real, fake)

