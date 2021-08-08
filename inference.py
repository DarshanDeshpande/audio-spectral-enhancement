from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
from pydub import AudioSegment
from tensorflow_io.python.api.audio import AudioIOTensor

from model import SingleBlock


def load_file(path1, audio_length):
    return AudioIOTensor(path1, dtype=tf.float32)[:audio_length]


def get_time_freq_data(audio_low, fft_length, frame_step, frame_length):
    mono1 = tf.math.reduce_mean(audio_low, axis=-1)
    ft1 = tf.signal.stft(
        signals=mono1,
        fft_length=fft_length,
        frame_step=frame_step,
        frame_length=frame_length,
    )

    return tf.stack((tf.math.real(ft1), tf.math.imag(ft1)), 0)


def convert_to_complex(pred):
    if pred.shape[0] == 1:
        pred = tf.squeeze(pred, 0)
    return tf.complex(pred[0, :, :], pred[1, :, :])


def get_time_series(audio_stft, fft_length, frame_step, frame_length):
    return tf.signal.inverse_stft(
        convert_to_complex(audio_stft), fft_length, frame_step, frame_length
    ).numpy()


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument(
        "--checkpoint-path", help="Path to model checkpoint", required=True
    )
    argparser.add_argument(
        "--mp3-audio-file",
        help="Path to MP3 audio file to be reconstructed",
        required=True,
    )
    argparser.add_argument(
        "--audio-length",
        help="Length of audio to be loaded that the model was trained on. Defaults to 100000 data points",
        default=100000,
        type=int,
    )
    argparser.add_argument(
        "--fft-length", help="FFT Length for STFT", default=1023, type=int
    )
    argparser.add_argument(
        "--frame-step", help="Frame Step for STFT", default=248, type=int
    )
    argparser.add_argument(
        "--frame-length", help="Frame Length for STFT", default=1024, type=int
    )
    argparser.add_argument(
        "--sampling-rate",
        help="Sampling rate of audio. Defaults to 22050Hz",
        default=22050,
        type=int,
    )
    parsed = argparser.parse_args()
    print("Loading the model")
    model = tf.keras.models.load_model(
        parsed.checkpoint_path,
        compile=False,
        custom_objects={"SingleBlock": SingleBlock},
    )
    print("Finished loading")

    print("Converting the audio wave")
    stft = get_time_freq_data(
        load_file(parsed.mp3_audio_file, parsed.audio_length),
        parsed.fft_length,
        parsed.frame_step,
        parsed.frame_length,
    )
    assert (
        stft.shape == model.input_shape[1:]
    ), f"Invalid input audio shape. The STFT input shape for the model should be {model.input_shape[1:]}"
    prediction = model(tf.expand_dims(stft, 0), training=False)[0]
    wave = get_time_series(
        tf.squeeze(prediction, 0),
        parsed.fft_length,
        parsed.frame_step,
        parsed.frame_length,
    )

    def write(f, sr, x, normalized=True):
        if normalized:
            y = np.int16(x * 2 ** 15)
        else:
            y = np.int16(x)
        song = AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=1)
        song.export(f, format="mp3", bitrate="128k")

    write("output.mp3", parsed.sampling_rate, wave, normalized=True)
    print("File written to output.mp3")
