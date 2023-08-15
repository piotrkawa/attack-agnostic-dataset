# Attack Agnostic Dataset: Towards Generalization and Stabilization of Audio DeepFake Detection

The following repository contains code for our paper called ["Attack Agnostic Dataset: Towards Generalization and Stabilization of Audio DeepFake Detection"](https://arxiv.org/abs/2206.13979).


We base our codebase on [WaveFake's repository](https://github.com/RUB-SysSec/WaveFake) (commit: `d52d51b`).



## Before you start


### Datasets

Download appropriate datasets:

* [ASVspoof2019 LA subset](https://datashare.ed.ac.uk/handle/10283/3336),
* [FakeAVCeleb](https://github.com/DASH-Lab/FakeAVCeleb#access-request-form),
* [WaveFake](https://zenodo.org/record/5642694) (along with JSUT and LJSpeech).

The above datasets result in a total of 286,014 samples used in training procedure.


### Dependencies
Install required dependencies using: 
```bash
pip install -r requirements.txt
```

### Configs

Both training and evaluation scripts are configured with the use of CLI and `.yaml` configuration files. File defines processing applied to raw audio files, as well as used architecture. An example config of LCNN architecture with LFCC frontend looks as follows:
```yaml
data:
  seed: 42
  cnn_features_setting:
    frontend_algorithm: ["lfcc"]  # ["mfcc"] or ["lfcc"] or ["mfcc", "lfcc"] or []
    use_spectrogram: False

checkpoint: 
  # This part is used only in evaluation (each checkpoint is used in eval on corresponding fold).
  # To ensure reliable results make sure that the order of checkpoints is correct (i.e. fold_0, fold_1, fold_2)
  paths: [
    "trained_models/aad__lcnn_fold_0/ckpt.pth",
    "trained_models/aad__lcnn_fold_1/ckpt.pth",
    "trained_models/aad__lcnn_fold_2/ckpt.pth",
  ]

model:
  name: "lcnn"  # {"rawnet", "mesonet_inception", "xception", "lcnn"}
  parameters:
    input_channels: 1  # 1 for each mfcc and lfcc, 2 for spec
  optimizer:
    lr: 0.0001
```


Other example configs are available under `configs/training/`

##  Train models 


To train models use `train_models.py`. It trains 3 models basing on Attack Agnostic Dataset. Each model is using different fold of the dataset.


```
usage: train_models.py [-h] [--asv_path ASV_PATH] [--wavefake_path WAVEFAKE_PATH] [--celeb_path CELEB_PATH] [--config CONFIG] [--amount AMOUNT] [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--ckpt CKPT] [--cpu] [--verbose] [--use_gmm] [--clusters CLUSTERS] [--lfcc]

optional arguments:
  -h, --help            show this help message and exit
  --asv_path ASV_PATH   Path to ASVspoof2021 dataset directory
  --wavefake_path WAVEFAKE_PATH
                        Path to WaveFake dataset directory
  --celeb_path CELEB_PATH
                        Path to FakeAVCeleb dataset directory
  --config CONFIG       Model config file path (default: config.yaml)
  --amount AMOUNT, -a AMOUNT
                        Amount of files to load - useful when debugging (default: None - use all).
  --batch_size BATCH_SIZE, -b BATCH_SIZE
                        Batch size (default: 128).
  --epochs EPOCHS, -e EPOCHS
                        Epochs (default: 5).
  --ckpt CKPT           Checkpoint directory (default: trained_models).
  --cpu, -c             Force using cpu?
  --verbose, -v         Display debug information?
  --use_gmm             [GMM] Use to train GMM, otherwise - NNs
  --clusters CLUSTERS, -k CLUSTERS
                        [GMM] The amount of clusters to learn (default: 128).
  --lfcc, -l            [GMM] Use LFCC instead of MFCC?
```

e.g. to train LCNN network use:
```bash
python train_models.py --asv_path ../datasets/ASVspoof2021/LA --wavefake_path ../datasets/WaveFake --celeb_path ../datasets/FakeAVCeleb/FakeAVCeleb_v1.2 --config configs/training/lcnn.yaml
```

To train GMM models:
```bash
python train_models.py --asv_path ../datasets/ASVspoof2021/LA --wavefake_path ../datasets/WaveFake --celeb_path ../datasets/FakeAVCeleb/FakeAVCeleb_v1.2 --lfcc --use_gmm
```

## Evaluate models


Once your models are trained you can evalaute them using `evaluate_models.py`.

**Before you start:** add checkpoint paths to the config used in training process.

**Note**: to make sure that results are realiable, paths should be provided in correct order (fold_1, fold_2, fold_3) - each checkpoint is evaluated on corresponding fold.


```
usage: evaluate_models.py [-h] [--asv_path ASV_PATH] [--wavefake_path WAVEFAKE_PATH] [--celeb_path CELEB_PATH] [--config CONFIG] [--amount AMOUNT] [--cpu] [--use_gmm] [--clusters CLUSTERS] [--lfcc] [--output OUTPUT] [--ckpt CKPT]

optional arguments:
  -h, --help            show this help message and exit
  --asv_path ASV_PATH
  --wavefake_path WAVEFAKE_PATH
  --celeb_path CELEB_PATH
  --config CONFIG       Model config file path (default: config.yaml)
  --amount AMOUNT, -a AMOUNT
                        Amount of files to load from each directory (default: None - use all).
  --cpu, -c             Force using cpu
  --use_gmm             [GMM] Use to evaluate GMM, otherwise - NNs
  --clusters CLUSTERS, -k CLUSTERS
                        [GMM] The amount of clusters to learn (default: 128).
  --lfcc, -l            [GMM] Use LFCC instead of MFCC?
  --output OUTPUT, -o OUTPUT
                        [GMM] Output file name.
  --ckpt CKPT           [GMM] Checkpoint directory (default: trained_models).
```
e.g. to evaluate LCNN network add appropriate checkpoint paths to config and then use:
```
python evaluate_models.py --config configs/training/lcnn.yaml --asv_path ../datasets/ASVspoof2021/LA --wavefake_path ../datasets/WaveFake --celeb_path ../datasets/FakeAVCeleb/FakeAVCeleb_v1.2
```

To evaluate GMM models:
```bash
python evaluate_models.py --asv_path ../datasets/ASVspoof2021/LA --wavefake_path ../datasets/WaveFake --celeb_path ../datasets/FakeAVCeleb/FakeAVCeleb_v1.2 --lfcc --use_gmm
```

## Citation

If you use this code in your research please use the following citation:

```
@inproceedings{kawa22_interspeech,
  author={Piotr Kawa and Marcin Plata and Piotr Syga},
  title={{Attack Agnostic Dataset: Towards Generalization and Stabilization of Audio DeepFake Detection}},
  year=2022,
  booktitle={Proc. Interspeech 2022},
  pages={4023--4027},
  doi={10.21437/Interspeech.2022-10078}
}
```


