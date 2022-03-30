from pathlib import Path

from moviepy.editor import *

FAKEAVCELEB_DATASET_PATH = ""  # path to FakeAVCeleb_v1.2
VIDEO_FOLDER = "FakeAVCeleb"
AUDIO_FOLDER = "FakeAVCeleb-audio"
FILE_EXTENSION = ".mp3"

source_path = Path(FAKEAVCELEB_DATASET_PATH)
mp4_files = list(source_path.glob("**/*.mp4"))

for file in mp4_files:

    print("source", file)

    relative_path = Path(file).relative_to(source_path / VIDEO_FOLDER)
    destination_path = source_path / AUDIO_FOLDER / relative_path
    destination_path = destination_path.with_suffix(FILE_EXTENSION)

    clip = AudioFileClip(str(file))
    destination_path.parent.mkdir(exist_ok=True, parents=True)
    clip.write_audiofile(str(destination_path))
