import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import torch
import yaml

from dfadetect.agnostic_datasets.attack_agnostic_dataset import AttackAgnosticDataset
from dfadetect.cnn_features import CNNFeaturesSetting
from dfadetect.datasets import apply_feature_and_double_delta, lfcc, mfcc
from dfadetect.models import models
from dfadetect.models.gaussian_mixture_model import GMMDescent, flatten_dataset
from dfadetect.trainer import GDTrainer, GMMTrainer, NNDataSetting
from dfadetect.utils import set_seed
from experiment_config import feature_kwargs

LOGGER = logging.getLogger()


def init_logger(log_file):
    LOGGER.setLevel(logging.INFO)

    # create file handler
    fh = logging.FileHandler(log_file)

    # # create console handler
    ch = logging.StreamHandler()

    # # create formatter and add it to the handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    LOGGER.addHandler(fh)
    LOGGER.addHandler(ch)


def save_model(
    model: torch.nn.Module,
    model_dir: Union[Path, str],
    name: str,
) -> None:
    full_model_dir = Path(f"{model_dir}/{name}")
    full_model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"{full_model_dir}/ckpt.pth")


def train_nn(
    datasets_paths: List[Union[Path, str]],
    batch_size: int,
    epochs: int,
    device: str,
    model_config: Dict,
    cnn_features_setting: CNNFeaturesSetting,
    model_dir: Optional[Path] = None,
    amount_to_use: Optional[int] = None,
) -> None:

    LOGGER.info("Loading data...")
    model_name, model_parameters = model_config["name"], model_config["parameters"]
    optimizer_config = model_config["optimizer"]

    use_cnn_features = False if model_name == "rawnet" else True

    nn_data_setting = NNDataSetting(
        use_cnn_features=use_cnn_features,
    )
    timestamp = time.time()
    folds_number = 3

    for fold in range(folds_number):
        data_train = AttackAgnosticDataset(
            asvspoof_path=datasets_paths[0],
            wavefake_path=datasets_paths[1],
            fakeavceleb_path=datasets_paths[2],
            fold_num=fold,
            fold_subset="train",
            reduced_number=amount_to_use,
            oversample=True,
        )

        data_test = AttackAgnosticDataset(
            asvspoof_path=datasets_paths[0],
            wavefake_path=datasets_paths[1],
            fakeavceleb_path=datasets_paths[2],
            fold_num=fold,
            fold_subset="test",
            reduced_number=amount_to_use,
            oversample=True,
        )

        current_model = models.get_model(
            model_name=model_name, config=model_parameters, device=device,
        ).to(device)

        LOGGER.info(f"Training '{model_name}' model on {len(data_train)} audio files.")

        current_model = GDTrainer(
            device=device,
            batch_size=batch_size,
            epochs=epochs,
            optimizer_kwargs=optimizer_config,
        ).train(
            dataset=data_train,
            model=current_model,
            test_dataset=data_test,
            nn_data_setting=nn_data_setting,
            logging_prefix=f"fold_{fold}",
            cnn_features_setting=cnn_features_setting,
        )

        if model_dir is not None:
            save_name = f"aad__{model_name}_fold_{fold}__{timestamp}"
            save_model(
                model=current_model,
                model_dir=model_dir,
                name=save_name,
            )
        LOGGER.info(f"Training model on fold [{fold+1}/{folds_number}] done!")


def train_gmm(
    datasets_paths: List[Union[Path, str]],
    feature_fn: Callable,
    feature_kwargs: dict,
    clusters: int,
    batch_size: int,
    device: str,
    model_dir: Optional[Path] = None,
    use_double_delta: bool = True,
    amount_to_use: Optional[int] = None,
    real_epochs: int = 3,
    fake_epochs: int = 1
) -> None:

    LOGGER.info("Loading data...")

    for fold in range(3):
        real_dataset_train = AttackAgnosticDataset(
            asvspoof_path=datasets_paths[0],
            wavefake_path=datasets_paths[1],
            fakeavceleb_path=datasets_paths[2],
            fold_num=fold,
            fold_subset="train",
            oversample=False,
            undersample=False,
            return_label=False,
            reduced_number=amount_to_use
        )
        real_dataset_train.get_bonafide_only()

        fake_dataset_train = AttackAgnosticDataset(
            asvspoof_path=datasets_paths[0],
            wavefake_path=datasets_paths[1],
            fakeavceleb_path=datasets_paths[2],
            fold_num=fold,
            fold_subset="train",
            oversample=False,
            undersample=False,
            return_label=False,
            reduced_number=amount_to_use
        )
        fake_dataset_train.get_spoof_only()

        real_dataset_train, fake_dataset_train = apply_feature_and_double_delta(
            [real_dataset_train, fake_dataset_train],
            feature_fn=feature_fn,
            feature_kwargs=feature_kwargs,
            use_double_delta=use_double_delta
        )

        LOGGER.info(f"GMM - Training real model on {len(real_dataset_train)} audio files.")
        inital_data = flatten_dataset(real_dataset_train, device, 10)
        real_model = GMMDescent(clusters, inital_data, covariance_type="diag").to(device)
        real_model = GMMTrainer(device=device, epochs=real_epochs, batch_size=batch_size).train(
            real_model, real_dataset_train, test_len=0.05
        )

        if model_dir is not None:
            save_model(
                model=real_model,
                model_dir=model_dir,
                name=f"real_{fold}",
            )
        LOGGER.info("Training real model done!")

        LOGGER.info(f"GMM - Training fake model on {len(fake_dataset_train)} audio files.")
        inital_data = flatten_dataset(fake_dataset_train, device, 10)
        fake_model = GMMDescent(clusters, inital_data, covariance_type="diag").to(device)
        fake_model = GMMTrainer(device=device, epochs=fake_epochs, batch_size=batch_size).train(
            fake_model, fake_dataset_train, test_len=.05
        )

        if model_dir is not None:
            save_model(
                model=fake_model,
                model_dir=model_dir,
                name=f"fake_{fold}",
            )

        LOGGER.info("Training fake model done!")


def main(args):
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    if args.verbose:
        LOGGER.setLevel(logging.INFO)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    seed = config["data"].get("seed", 42)
    # fix all seeds
    set_seed(seed)

    if not args.cpu and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model_dir = Path(args.ckpt)
    model_dir.mkdir(parents=True, exist_ok=True)

    if not args.use_gmm:
        cnn_features_setting = config["data"].get("cnn_features_setting", None)
        if cnn_features_setting:
            cnn_features_setting = CNNFeaturesSetting(**cnn_features_setting)
        else:
            cnn_features_setting = CNNFeaturesSetting()

        train_nn(
            datasets_paths=[args.asv_path, args.wavefake_path, args.celeb_path],
            device=device,
            amount_to_use=args.amount,
            batch_size=args.batch_size,
            epochs=args.epochs,
            model_dir=model_dir,
            model_config=config["model"],
            cnn_features_setting=cnn_features_setting,
        )
    else:
        feature_fn = lfcc if args.lfcc else mfcc
        train_gmm(
            datasets_paths=[args.asv_path, args.wavefake_path, args.celeb_path],
            feature_fn=feature_fn,
            feature_kwargs=feature_kwargs(args.lfcc),
            clusters=args.clusters,
            batch_size=args.batch_size,
            device=device,
            model_dir=model_dir,
            use_double_delta=True,
            amount_to_use=args.amount
        )


def parse_args():
    parser = argparse.ArgumentParser()

    ASVSPOOF_DATASET_PATH = "../datasets/ASVspoof2021/LA"
    WAVEFAKE_DATASET_PATH = "../datasets/WaveFake"
    FAKEAVCELEB_DATASET_PATH = "../datasets/FakeAVCeleb/FakeAVCeleb_v1.2"

    parser.add_argument(
        "--asv_path", type=str, default=ASVSPOOF_DATASET_PATH, help="Path to ASVspoof2021 dataset directory",
    )
    parser.add_argument(
        "--wavefake_path", type=str, default=WAVEFAKE_DATASET_PATH, help="Path to WaveFake dataset directory",
    )
    parser.add_argument(
        "--celeb_path", type=str, default=FAKEAVCELEB_DATASET_PATH, help="Path to FakeAVCeleb dataset directory",
    )

    default_model_config = "config.yaml"
    parser.add_argument(
        "--config", help="Model config file path (default: config.yaml)", type=str, default=default_model_config
    )

    default_amount = None
    parser.add_argument(
        "--amount", "-a", help=f"Amount of files to load - useful when debugging (default: {default_amount} - use all).", type=int, default=default_amount
    )

    default_batch_size = 128
    parser.add_argument(
        "--batch_size", "-b", help=f"Batch size (default: {default_batch_size}).", type=int, default=default_batch_size)

    default_epochs = 5
    parser.add_argument(
        "--epochs", "-e", help=f"Epochs (default: {default_epochs}).", type=int, default=default_epochs)

    default_model_dir = "trained_models"
    parser.add_argument(
        "--ckpt", help=f"Checkpoint directory (default: {default_model_dir}).", type=str, default=default_model_dir)

    parser.add_argument(
        "--cpu", "-c", help="Force using cpu?", action="store_true")

    parser.add_argument(
        "--verbose", "-v", help="Display debug information?", action="store_true")

    # GMM arguments
    parser.add_argument(
        "--use_gmm", help="[GMM] Use to train GMM, otherwise - NNs", action="store_true"
    )

    default_k = 128
    parser.add_argument(
        "--clusters", "-k", help=f"[GMM] The amount of clusters to learn (default: {default_k}).", type=int, default=default_k)

    parser.add_argument(
        "--lfcc", "-l", help="[GMM] Use LFCC instead of MFCC?", action="store_true"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
