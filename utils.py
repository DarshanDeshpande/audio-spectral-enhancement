import glob

import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_io as tfio
from sklearn.model_selection import train_test_split

SEED = 32


class DatasetLoader:
    def __init__(
        self,
        audio_low_path,
        audio_high_path,
        audio_length,
        fft_length=1023,
        frame_step=248,
        frame_length=1024,
        cache=False,
    ):
        self.audio_length = audio_length
        self.fft_length = fft_length
        self.frame_step = frame_step
        self.frame_length = frame_length
        self.audio_low_path = audio_low_path
        self.audio_high_path = audio_high_path
        self.cache = cache

    def load_file(self, path1, path2):
        return (
            tfio.audio.AudioIOTensor(path1, dtype=tf.float32)[: self.audio_length],
            tfio.audio.AudioIOTensor(path2, dtype=tf.float32)[: self.audio_length],
        )

    def get_time_freq_data(self, audio_low, audio_high):
        # Output shape = (2, Frames, Freq bins)
        # Frames = (samples-frame_length)/hop_size = (100000-1024)/248 = 400
        # Freq bins = (fft_length/2)+1 = (1023/2)+1 = 512
        mono1 = tf.math.reduce_mean(audio_low, axis=-1)
        ft1 = tf.signal.stft(
            signals=mono1,
            fft_length=self.fft_length,
            frame_step=self.frame_step,
            frame_length=self.frame_length,
        )

        mono2 = tf.math.reduce_mean(audio_high, axis=-1)
        ft2 = tf.signal.stft(
            signals=mono2,
            fft_length=self.fft_length,
            frame_step=self.frame_step,
            frame_length=self.frame_length,
        )

        return tf.stack((tf.math.real(ft1), tf.math.imag(ft1)), 0), tf.stack(
            (tf.math.real(ft2), tf.math.imag(ft2)), 0
        )

    def train_val_test_split(self):
        # Splitting the Dataset
        low_glob = sorted(glob.glob(self.audio_low_path))
        high_glob = sorted(glob.glob(self.audio_high_path))

        X_train, X_val, Y_train, Y_val = train_test_split(
            low_glob, high_glob, test_size=0.04, random_state=SEED
        )
        X_val, X_test, Y_val, Y_test = train_test_split(
            X_val, Y_val, test_size=0.3, random_state=SEED
        )
        print(
            f"Number of Training samples: {len(X_train)}, "
            f"Number of Training samples: {len(X_val)}, "
            f"Number of Training samples: {len(X_test)}"
        )

        return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)

    def load(self, train_tuple, val_tuple, test_tuple):
        (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = (
            train_tuple,
            val_tuple,
            test_tuple,
        )

        dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        loaded_dataset = dataset.map(self.load_file, tf.data.experimental.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
        loaded_val_dataset = val_dataset.map(
            self.load_file, tf.data.experimental.AUTOTUNE
        )

        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
        loaded_test_dataset = test_dataset.map(
            self.load_file, tf.data.experimental.AUTOTUNE
        )

        if self.cache:
            tfd_dataset = loaded_dataset.map(
                self.get_time_freq_data, tf.data.experimental.AUTOTUNE
            ).cache("t-map")
            tfd_val_dataset = loaded_val_dataset.map(
                self.get_time_freq_data, tf.data.experimental.AUTOTUNE
            ).cache("t-val-map")
            tfd_test_dataset = loaded_test_dataset.map(
                self.get_time_freq_data, tf.data.experimental.AUTOTUNE
            ).cache("t-test-map")
        else:
            tfd_dataset = loaded_dataset.map(
                self.get_time_freq_data, tf.data.experimental.AUTOTUNE
            )
            tfd_val_dataset = loaded_val_dataset.map(
                self.get_time_freq_data, tf.data.experimental.AUTOTUNE
            )
            tfd_test_dataset = loaded_test_dataset.map(
                self.get_time_freq_data, tf.data.experimental.AUTOTUNE
            )

        return tfd_dataset, tfd_val_dataset, tfd_test_dataset


# Custom Callback for visualizing samples during training
class SpecPlotCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        X_test: list,
        Y_test: list,
        fft_length=1023,
        frame_step=248,
        frame_length=1024,
        *args,
        **kwargs,
    ):
        super(SpecPlotCallback, self).__init__(*args, **kwargs)
        a1, a2 = self.load_file(X_test[5], Y_test[5])
        i, j = self.get_time_freq_data(a1, a2)
        # self.i, self.j = i * (fft_length / 2), j * (512 / 2)
        print(self.i, self.j)

        self.fft_length = fft_length
        self.frame_step = frame_step
        self.frame_length = frame_length
        self.audio_length = 100000

    def load_file(self, path1, path2):
        return (
            tfio.audio.AudioIOTensor(path1, dtype=tf.float32)[: self.audio_length],
            tfio.audio.AudioIOTensor(path2, dtype=tf.float32)[: self.audio_length],
        )

    def get_time_freq_data(self, audio_low, audio_high):
        mono1 = tf.math.reduce_mean(audio_low, axis=-1)
        ft1 = tf.signal.stft(
            signals=mono1,
            fft_length=self.fft_length,
            frame_step=self.frame_step,
            frame_length=self.frame_length,
        )

        mono2 = tf.math.reduce_mean(audio_high, axis=-1)
        ft2 = tf.signal.stft(
            signals=mono2,
            fft_length=self.fft_length,
            frame_step=self.frame_step,
            frame_length=self.frame_length,
        )

        return tf.stack((tf.math.real(ft1), tf.math.imag(ft1)), 0), tf.stack(
            (tf.math.real(ft2), tf.math.imag(ft2)), 0
        )

    def convert_to_complex(self, pred):
        if pred.shape[0] == 1:
            pred = tf.squeeze(pred, 0)
        return tf.complex(pred[0, :, :], pred[1, :, :])

    def plot(self):
        query = tf.expand_dims(self.i, 0)
        pred = self.model.predict(query)[0]
        pred = pred * (512 / 2)
        plt.figure(figsize=(20, 4.5))
        plt.subplot(131)
        librosa.display.specshow(
            librosa.amplitude_to_db(
                np.abs(self.convert_to_complex(self.i).numpy().T), ref=np.max
            ),
            x_axis="time",
            y_axis="linear",
            sr=22050,
        )
        plt.subplot(132)
        librosa.display.specshow(
            librosa.amplitude_to_db(
                np.abs(self.convert_to_complex(self.j).numpy().T), ref=np.max
            ),
            x_axis="time",
            y_axis="linear",
            sr=22050,
        )
        plt.subplot(133)
        librosa.display.specshow(
            librosa.amplitude_to_db(
                np.abs(self.convert_to_complex(pred).numpy().T), ref=np.max
            ),
            x_axis="time",
            y_axis="linear",
            sr=22050,
        )
        plt.show()

        plt.subplot(121)
        _ = plt.magnitude_spectrum(
            tf.signal.inverse_stft(
                self.convert_to_complex(self.j),
                fft_length=self.fft_length,
                frame_step=self.frame_step,
                frame_length=self.frame_length,
            ).numpy(),
            color="black",
            sides="onesided",
            scale="dB",
        )[0]
        _ = plt.magnitude_spectrum(
            tf.signal.inverse_stft(
                self.convert_to_complex(pred),
                fft_length=self.fft_length,
                frame_step=self.frame_step,
                frame_length=self.frame_length,
            ).numpy(),
            color="orange",
            sides="onesided",
            scale="dB",
        )[0]
        plt.legend(["Actual", "Prediction"])

        plt.subplot(122)
        _ = plt.phase_spectrum(
            tf.signal.inverse_stft(
                self.convert_to_complex(self.j),
                fft_length=self.fft_length,
                frame_step=self.frame_step,
                frame_length=self.frame_length,
            ).numpy(),
            color="black",
            sides="onesided",
        )[0]
        _ = plt.phase_spectrum(
            tf.signal.inverse_stft(
                self.convert_to_complex(pred),
                fft_length=self.fft_length,
                frame_step=self.frame_step,
                frame_length=self.frame_length,
            ).numpy(),
            color="orange",
            sides="onesided",
        )[0]
        _ = plt.phase_spectrum(
            tf.signal.inverse_stft(
                self.convert_to_complex(self.i),
                fft_length=self.fft_length,
                frame_step=self.frame_step,
                frame_length=self.frame_length,
            ).numpy(),
            color="red",
            sides="onesided",
        )[0]
        plt.legend(["Actual", "Prediction", "Low"])
        plt.show()

    def on_epoch_end(self, epoch, logs=None):
        self.plot()


# class PixelLoss(tf.keras.losses.Loss):
#     def __init__(self, *args, **kwargs):
#         super(PixelLoss, self).__init__(*args, **kwargs)

#     def loss(self, y_true, y_pred):
#         total_error = K.sum(K.square(y_true - K.mean(y_true)))
#         residual_error = K.sum(K.square(y_true - y_pred))
#         R_squared = 1 - (residual_error / total_error)
#         return -R_squared

#     def call(self, y_true, y_pred):
#       return 1+self.loss(y_true,y_pred)


def r2_score(y_true, y_pred):
    total_error = K.sum(K.square(y_true - K.mean(y_true)))
    residual_error = K.sum(K.square(y_true - y_pred))
    R_squared = 1 - (residual_error / total_error)
    return -R_squared


def PixelLoss(y_true, y_pred):
    return 1 + r2_score(y_true, y_pred)


class SSIM(tf.keras.losses.Loss):
    def __init__(self, filter_size=3, max_val=1, **kwargs):
        super(SSIM, self).__init__(**kwargs)
        self.filter_size = filter_size
        self.max_val = max_val

    def ssim_loss(self, y_true, y_pred, filter_size=3, max_val=1):
        return 1 - (
            tf.reduce_sum(
                tf.image.ssim(y_true, y_pred, max_val=max_val, filter_size=filter_size)
            )
        ) / tf.cast(tf.shape(y_pred)[0], tf.float32)

    def call(self, y_true, y_pred):
        true_real, true_imag = y_true[:, 0, :, :], y_true[:, 1, :, :]
        pred_real, pred_imag = y_pred[:, 0, :, :], y_pred[:, 1, :, :]
        complex_true = tf.complex(true_real, true_imag)
        complex_pred = tf.complex(pred_real, pred_imag)

        return self.ssim_loss(
            tf.expand_dims(tf.math.abs(complex_true), -1),
            tf.expand_dims(tf.math.abs(complex_pred), -1),
            filter_size=self.filter_size,
            max_val=self.max_val,
        )
