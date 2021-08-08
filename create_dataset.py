import glob
import os
import shutil
import subprocess

import scipy.io.wavfile as wavf
import tensorflow as tf
import tensorflow_datasets as tfds
from pydub import AudioSegment
from tqdm import tqdm

os.mkdir("WavFiles")
os.mkdir("FFMPEGConverted")
os.mkdir("CutSounds128")
os.mkdir("FFMPEGConvertedCut32")

dataset = tfds.load("gtzan")

for index, i in tqdm(enumerate(dataset["train"])):
    wavf.write(
        f"WavFiles/{index}.wav",
        22050,
        (tf.cast(i["audio"], tf.float32) / 32768.0).numpy(),
    )

print("Converting WAV files to 128Kbps MP3 files")
for name in tqdm(glob.glob("WavFiles/*")):
    mp4 = '"%s" -codec:a libmp3lame -b:a 128K -vn ' % name
    mp3 = '"FFMPEGConverted/%s.mp3"' % os.path.basename(name).split(".")[0]
    ffmpeg = """ffmpeg -i %s""" % mp4 + mp3
    subprocess.call(ffmpeg, shell=True)

print("Splitting audio files into 3 parts for augmentation")
for i in tqdm(glob.glob("FFMPEGConverted/*")):
    sound = AudioSegment.from_mp3(i)
    cut = len(sound) // 3

    for index, j in enumerate(range(0, cut * 3, cut)):
        f = sound[j : j + cut]
        f.export(
            f"CutSounds128/{index + 1}_{i.split('/')[-1].split('.')[0]}.mp3",
            format="mp3",
            bitrate="128k",
            codec="libmp3lame",
        )

print("Converting 128Kbps to 32Kbps")
for name in tqdm(glob.glob("CutSounds128/*")):
    mp4 = '"%s" -codec:a libmp3lame -b:a 32K -vn ' % name
    mp3 = '"FFMPEGConvertedCut32/%s.mp3"' % os.path.basename(name).split(".")[0]
    ffmpeg = """ffmpeg -i %s""" % mp4 + mp3
    subprocess.call(ffmpeg, shell=True)

print("Cleaning up")
shutil.rmtree("FFMPEGConverted")
shutil.rmtree("WavFiles")
print("Finished converting.")
print(f"High bitrate audio files: CutSounds128")
print(f"Low bitrate audio files: FFMPEGConvertedCut32")
