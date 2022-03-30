import argparse
import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import yaml
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import (auc, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)
from torch.utils.data import DataLoader

from dfadetect import cnn_features
from dfadetect.agnostic_datasets.attack_agnostic_dataset import \
    AttackAgnosticDataset
from dfadetect.cnn_features import CNNFeaturesSetting
from dfadetect.datasets import (TransformDataset,
                                apply_feature_and_double_delta, lfcc, mfcc)
from dfadetect.models import models
from dfadetect.models.gaussian_mixture_model import (GMMBase, classify_dataset,
                                                     load_model)
from dfadetect.trainer import NNDataSetting
from dfadetect.utils import set_seed
from experiment_config import feature_kwargs

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)

ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
LOGGER.addHandler(ch)



def plot_roc(
    fpr: np.ndarray,
    tpr: np.ndarray,
    training_dataset_name: str,
    fake_dataset_name: str,
    path: str,
    lw: int = 2,
    save: bool = False,
) -> matplotlib.figure.Figure:
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    # ax.set_title(
        # f'Train: {training_dataset_name}\nEvaluated on {fake_dataset_name}')
    ax.legend(loc="lower right")

    fig.tight_layout()
    if save:
        fig.savefig(f"{path}.pdf")
    plt.close(fig)
    return fig


def calculate_eer(y, y_score) -> Tuple[float, float, np.ndarray, np.ndarray]:
    fpr, tpr, thresholds = roc_curve(y, -y_score)

    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return thresh, eer, fpr, tpr


def calculate_eer_for_models(
    real_model: GMMBase,
    fake_model: GMMBase,
    real_dataset_test: TransformDataset,
    fake_dataset_test: TransformDataset,
    training_dataset_name: str,
    fake_dataset_name: str,
    plot_dir_path: str,
    device: str,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    real_scores = classify_dataset(
        real_model,
        fake_model,
        real_dataset_test,
        device
    ).numpy()

    fake_scores = classify_dataset(
        real_model,
        fake_model,
        fake_dataset_test,
        device
    ).numpy()

    # JSUT fake samples are fewer available
    length = min(len(real_scores),  len(fake_scores))
    real_scores = real_scores[:length]
    fake_scores = fake_scores[:length]

    labels = np.concatenate(
        (
            np.zeros(real_scores.shape, dtype=np.int32),
            np.ones(fake_scores.shape, dtype=np.int32)
        )
    )

    thresh, eer, fpr, tpr = calculate_eer(
        y=np.array(labels, dtype=np.int32),
        y_score=np.concatenate((real_scores, fake_scores)),
    )

    fig_path = f"{plot_dir_path}/{training_dataset_name.replace('.', '_').replace('/', '_')}_{fake_dataset_name.replace('.', '_').replace('/', '_')}"
    plot_roc(fpr, tpr, training_dataset_name, fake_dataset_name, fig_path)

    return eer, thresh, fpr, tpr


def evaluate_nn(
    model_paths: List[Path],
    datasets_paths: List[Union[Path, str]],
    data_config: Dict,
    model_config: Dict,
    device: str,
    amount_to_use: Optional[int] = None,
    batch_size: int = 128,
):
    LOGGER.info("Loading data...")
    model_name, model_parameters = model_config["name"], model_config["parameters"]
    use_cnn_features = False if model_name == "rawnet" else True
    cnn_features_setting = data_config.get("cnn_features_setting", None)

    nn_data_setting = NNDataSetting(
        use_cnn_features=use_cnn_features,
    )

    if use_cnn_features:
        cnn_features_setting = CNNFeaturesSetting(**cnn_features_setting)
    else:
        cnn_features_setting = CNNFeaturesSetting()

    weights_path = ''
    for fold in tqdm.tqdm(range(3)):
        # Load model architecture
        model = models.get_model(
            model_name=model_name, config=model_parameters, device=device,
        )
        # If provided weights, apply corresponding ones (from an appropriate fold)
        if len(model_paths) > 1:
            assert len(model_paths) == 3, "Pass either 0 or 3 weights path"
            weights_path = model_paths[fold]
            model.load_state_dict(
                torch.load(weights_path)
            )
        model = model.to(device)

        logging_prefix = f"fold_{fold}"
        data_val = AttackAgnosticDataset(
            asvspoof_path=datasets_paths[0],
            wavefake_path=datasets_paths[1],
            fakeavceleb_path=datasets_paths[2],
            fold_num=fold,
            fold_subset="val",
            reduced_number=amount_to_use,
        )
        LOGGER.info(f"Testing '{model_name}' model, weights path: '{weights_path}', on {len(data_val)} audio files.")
        print(f"Test Fold [{fold+1}/{3}]: ")
        test_loader = DataLoader(
            data_val,
            batch_size=batch_size,
            drop_last=True,
            num_workers=3,
        )

        num_correct = 0.0
        num_total = 0.0
        y_pred = torch.Tensor([]).to(device)
        y = torch.Tensor([]).to(device)
        y_pred_label = torch.Tensor([]).to(device)
        batches_number = len(data_val) // batch_size

        for i, (batch_x, _, batch_y) in enumerate(test_loader):
            model.eval()
            if i % 10 == 0:
                print(f"Batch [{i}/{batches_number}]")

            with torch.no_grad():
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                num_total += batch_x.size(0)

                if nn_data_setting.use_cnn_features:
                    batch_x = cnn_features.prepare_feature_vector(batch_x, cnn_features_setting=cnn_features_setting)

                batch_pred = model(batch_x).squeeze(1)
                batch_pred = torch.sigmoid(batch_pred)
                batch_pred_label = (batch_pred + .5).int()

                num_correct += (batch_pred_label == batch_y.int()).sum(dim=0).item()

                y_pred = torch.concat([y_pred, batch_pred], dim=0)
                y_pred_label = torch.concat([y_pred_label, batch_pred_label], dim=0)
                y = torch.concat([y, batch_y], dim=0)

        eval_accuracy = (num_correct / num_total) * 100

        precision, recall, f1_score, support = precision_recall_fscore_support(
            y.cpu().numpy(),
            y_pred_label.cpu().numpy(),
            average="binary",
            beta=1.0
        )
        auc_score = roc_auc_score(y_true=y.cpu().numpy(), y_score=y_pred_label.cpu().numpy())

        # For EER flip values, following original evaluation implementation
        y_for_eer = 1 - y

        thresh, eer, fpr, tpr = calculate_eer(
            y=y_for_eer.cpu().numpy(),
            y_score=y_pred.cpu().numpy(),
        )

        eer_label = f"eval/{logging_prefix}__eer"
        accuracy_label = f"eval/{logging_prefix}__accuracy"
        precision_label = f"eval/{logging_prefix}__precision"
        recall_label = f"eval/{logging_prefix}__recall"
        f1_label = f"eval/{logging_prefix}__f1_score"
        auc_label = f"eval/{logging_prefix}__auc"

        LOGGER.info(
            f"{eer_label}: {eer:.4f}, {accuracy_label}: {eval_accuracy:.4f}, {precision_label}: {precision:.4f}, {recall_label}: {recall:.4f}, {f1_label}: {f1_score:.4f}, {auc_label}: {auc_score:.4f}"
        )


def evaluate_gmm(
    real_model_path: str,
    fake_model_path: str,
    datasets_paths: List[str],
    amount_to_use: Optional[int],
    feature_fn: Callable,
    feature_kwargs: dict,
    clusters: int,
    device: str,
    frontend: str,
    output_file_name: str,
    use_double_delta: bool = True
):

    complete_results = {}

    LOGGER.info(f"paths: {real_model_path}, {fake_model_path}, {datasets_paths}")

    for subtype in ["val", "test", "train"]:
        for fold in [0, 1, 2]:
            real_dataset_test = AttackAgnosticDataset(
                asvspoof_path=datasets_paths[0],
                wavefake_path=datasets_paths[1],
                fakeavceleb_path=datasets_paths[2],
                fold_num=fold,
                fold_subset=subtype,
                oversample=False,
                undersample=False,
                return_label=False,
                reduced_number=amount_to_use,
            )
            real_dataset_test.get_bonafide_only()

            fake_dataset_test = AttackAgnosticDataset(
                asvspoof_path=datasets_paths[0],
                wavefake_path=datasets_paths[1],
                fakeavceleb_path=datasets_paths[2],
                fold_num=fold,
                fold_subset=subtype,
                oversample=False,
                undersample=False,
                return_label=False,
                reduced_number=amount_to_use,
            )
            fake_dataset_test.get_spoof_only()

            real_dataset_test, fake_dataset_test = apply_feature_and_double_delta(
                [real_dataset_test, fake_dataset_test],
                feature_fn=feature_fn,
                feature_kwargs=feature_kwargs,
                use_double_delta=use_double_delta
            )

            model_path = Path(real_model_path) / f"real_{fold}" / "ckpt.pth"
            real_model = load_model(
                real_dataset_test,
                str(model_path),
                device,
                clusters,
            )

            model_path = Path(fake_model_path) / f"fake_{fold}" / "ckpt.pth"
            fake_model = load_model(
                fake_dataset_test,
                str(model_path),
                device,
                clusters,
            )

            plot_path = Path(f"plots/{frontend}/fold_{fold}/{subtype}")
            if not plot_path.exists():
                plot_path.mkdir(parents=True)

            plot_path = str(plot_path)

            results = {"fold": fold}

            LOGGER.info(f"Calculating on folds...")

            eer, thresh, fpr, tpr = calculate_eer_for_models(
                real_model,
                fake_model,
                real_dataset_test,
                fake_dataset_test,
                f"train_fold_{fold}",
                "all",
                plot_dir_path=plot_path,
                device=device,
            )
            results["eer"] = str(eer)
            results["thresh"] = str(thresh)
            results["fpr"] = str(list(fpr))
            results["tpr"] = str(list(tpr))

            LOGGER.info(f"{subtype} | Fold {fold}:\n\tEER: {eer} Thresh: {thresh}")

            complete_results[subtype] = {}
            complete_results[subtype][fold] = results

    with open(f"{output_file_name}.json", "w+") as json_file:
        json.dump(complete_results, json_file, indent=4)


def main(args):

    if not args.cpu and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    seed = config["data"].get("seed", 42)
    # fix all seeds - this should not actually change anything
    set_seed(seed)

    if not args.use_gmm:
        evaluate_nn(
            model_paths=config["checkpoint"].get("paths", []),
            datasets_paths=[args.asv_path, args.wavefake_path, args.celeb_path],
            model_config=config["model"],
            data_config=config["data"],
            amount_to_use=args.amount,
            device=device,
        )
    else:
        evaluate_gmm(
            real_model_path=args.ckpt,
            fake_model_path=args.ckpt,
            datasets_paths=[args.asv_path, args.wavefake_path, args.celeb_path],
            feature_fn=lfcc if args.lfcc else mfcc,
            feature_kwargs=feature_kwargs(lfcc=args.lfcc),
            clusters=args.clusters,
            device=device,
            frontend="lfcc" if args.lfcc else "mfcc",
            amount_to_use=args.amount,
            output_file_name="gmm_evaluation",
            use_double_delta=True
        )


def parse_args():
    parser = argparse.ArgumentParser()

    # If assigned as None, then it won't be taken into account
    ASVSPOOF_DATASET_PATH = "../datasets/ASVspoof2021/LA"
    WAVEFAKE_DATASET_PATH = "../datasets/WaveFake"
    FAKEAVCELEB_DATASET_PATH = "../datasets/FakeAVCeleb/FakeAVCeleb_v1.2"

    parser.add_argument(
        "--asv_path", type=str, default=ASVSPOOF_DATASET_PATH
    )
    parser.add_argument(
        "--wavefake_path", type=str, default=WAVEFAKE_DATASET_PATH
    )
    parser.add_argument(
        "--celeb_path", type=str, default=FAKEAVCELEB_DATASET_PATH
    )

    default_model_config = "config.yaml"
    parser.add_argument(
        "--config", help="Model config file path (default: config.yaml)", type=str, default=default_model_config)

    default_amount = None
    parser.add_argument(
        "--amount", "-a", help=f"Amount of files to load from each directory (default: {default_amount} - use all).", type=int, default=default_amount)

    parser.add_argument(
        "--cpu", "-c", help="Force using cpu", action="store_true")

    parser.add_argument(
        "--use_gmm", help="[GMM] Use to evaluate GMM, otherwise - NNs", action="store_true"
    )

    default_k = 128
    parser.add_argument(
        "--clusters", "-k", help=f"[GMM] The amount of clusters to learn (default: {default_k}).", type=int, default=default_k
    )
    parser.add_argument(
        "--lfcc", "-l", help="[GMM] Use LFCC instead of MFCC?", action="store_true"
    )

    parser.add_argument(
        "--output", "-o", help="[GMM] Output file name.", type=str, default="results"
    )

    default_model_dir = "trained_models"
    parser.add_argument(
        "--ckpt", help=f"[GMM] Checkpoint directory (default: {default_model_dir}).", type=str, default=default_model_dir)

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
