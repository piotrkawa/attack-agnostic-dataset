from dataclasses import dataclass, field
from typing import List

import torch
import torchaudio


@dataclass
class CNNFeaturesSetting:
    frontend_algorithm: List[str]= field(default_factory=lambda: ["mfcc"])
    use_spectrogram: bool = True


# values from FakeAVCeleb paper
SAMPLING_RATE = 16_000
win_length = 400  # int((25 / 1_000) * SAMPLING_RATE)
hop_length = 160  # int((10 / 1_000) * SAMPLING_RATE)

device = "cuda" if torch.cuda.is_available() else "cpu"

MFCC_FN = torchaudio.transforms.MFCC(
    sample_rate=SAMPLING_RATE,
    n_mfcc=80,
    melkwargs={
        "n_fft": 512,
        "win_length": win_length,
        "hop_length": hop_length,
    },
).to(device)


LFCC_FN = torchaudio.transforms.LFCC(
    sample_rate=SAMPLING_RATE,
    n_lfcc=80,
    speckwargs={
        "n_fft": 512,
        "win_length": win_length,
        "hop_length": hop_length,
    },
).to(device)

MEL_SCALE_FN = torchaudio.transforms.MelScale(
    n_mels=80,
    n_stft=257,
    sample_rate=SAMPLING_RATE,
).to(device)


def prepare_feature_vector(
    audio: torch.Tensor,
    cnn_features_setting: CNNFeaturesSetting,
    win_length: int = 400,
    hop_length: int = 160,
) -> torch.Tensor:

    feature_vector = []

    if "mfcc" in cnn_features_setting.frontend_algorithm:
        mfcc_feature = MFCC_FN(audio)
        feature_vector.append(mfcc_feature)

    if "lfcc" in cnn_features_setting.frontend_algorithm:
        lfcc_feature = LFCC_FN(audio)
        feature_vector.append(lfcc_feature)

    if cnn_features_setting.use_spectrogram:
        stft_features = prepare_stft_features(audio, win_length, hop_length)
        feature_vector += stft_features  # abs_mel, abs_angle

    assert len(feature_vector) >= 1, "Feature vector must contain at least one feature!"

    feature_vector = torch.stack(feature_vector, dim=1)

    # [batch_size, feature_num, 80, frames], where feature_num in {1,2,3,4}
    return feature_vector


def prepare_stft_features(audio, win_length, hop_length):
    # Run STFT
    stft_out = torch.stft(
        audio,
        n_fft=512,
        return_complex=True,
        hop_length=hop_length,
        win_length=win_length,
    )

    # Reduce dimensionality via use of mel filterbanks
    stft_real_mel = MEL_SCALE_FN(stft_out.real)
    stft_imag_mel = MEL_SCALE_FN(stft_out.imag)

    complex_tensor = torch.complex(stft_real_mel, stft_imag_mel)
    stft_abs_mel = complex_tensor.abs()
    stft_abs_angle = complex_tensor.angle()
    return stft_abs_mel, stft_abs_angle
