import glob
from argparse import ArgumentParser

import tensorflow as tf
from sklearn.model_selection import train_test_split

from model import create_model
from utils import DatasetLoader, SSIM, PixelLoss

SEED = 32


def get_model(shape, num_blocks):
    return create_model(input_shape=shape, num_blocks=num_blocks)


def get_dataset(
    audio_low_path,
    audio_high_path,
    audio_length,
    fft_length=1023,
    frame_step=248,
    frame_length=1024,
    cache=True,
):
    return DatasetLoader(
        audio_low_path,
        audio_high_path,
        audio_length,
        fft_length=fft_length,
        frame_step=frame_step,
        frame_length=frame_length,
        cache=cache,
    )


def train_val_test_split(audio_low_path, audio_high_path, num_audio_files):
    # Splitting the Dataset
    low_glob = sorted(glob.glob(audio_low_path + "/*"))[:num_audio_files]
    high_glob = sorted(glob.glob(audio_high_path + "/*"))[:num_audio_files]

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


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument(
        "--audio-low-path",
        help="Low bitrate audio path. Use create_dataset.py to create the dataset if you haven't yet",
        required=True,
    )
    argparser.add_argument(
        "--audio-high-path",
        help="High bitrate audio path. Use create_dataset.py to create the dataset if you haven't yet",
        required=True,
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
        "--audio-length",
        help="Length of audio to be processed",
        default=100000,
        type=int,
    )
    argparser.add_argument(
        "--num-blocks",
        help="Number of Separable Convolution Blocks",
        default=1,
        type=int,
    )
    argparser.add_argument(
        "--learning-rate", help="Learning Rate for ADAM", default=1e-3, type=float
    )
    argparser.add_argument(
        "--batch-size", help="Batch size for training", default=32, type=int
    )
    argparser.add_argument("--epochs", help="Training epochs", default=200, type=int)
    argparser.add_argument("--cache", help="Training epochs", default=True, type=bool)
    argparser.add_argument(
        "--num-audio-files", help="Training epochs", default=1000, type=int
    )
    parsed = argparser.parse_args()

    train_tuple, val_tuple, test_tuple = train_val_test_split(
        parsed.audio_low_path, parsed.audio_high_path, parsed.num_audio_files
    )
    train_ds, val_ds, test_ds = get_dataset(
        parsed.audio_low_path,
        parsed.audio_high_path,
        parsed.audio_length,
        parsed.fft_length,
        parsed.frame_step,
        parsed.frame_length,
        cache=parsed.cache,
    ).load(train_tuple, val_tuple, test_tuple)
    print("Fetching input shape")
    input_shape = next(iter(train_ds))[0].shape
    print(f"Input shape according to your configuration: {input_shape}")

    model = get_model(input_shape, parsed.num_blocks)
    model.summary()

    model.compile(
        tf.keras.optimizers.Adam(parsed.learning_rate),
        loss=[PixelLoss, SSIM(reduction=tf.keras.losses.Reduction.NONE)],
        loss_weights=[1, 0.5],
    )
    history = model.fit(
        train_ds.shuffle(300)
        .batch(parsed.batch_size)
        .prefetch(tf.data.experimental.AUTOTUNE),
        epochs=parsed.epochs,
        validation_data=val_ds.batch(parsed.batch_size).prefetch(
            tf.data.experimental.AUTOTUNE
        ),
        callbacks=[
            # SpecPlotCallback(test_tuple[0], test_tuple[1], parsed.fft_length, parsed.frame_step,
            #                  parsed.frame_length),
            tf.keras.callbacks.ModelCheckpoint(
                "Checkpoints/{epoch}.h5", save_best_only=True, verbose=1
            )
        ],
    )
    print("-" * 100)
    print("Training finished")
    print("-" * 100)

    print("Starting evaluation on test set")
    print("Evaluation finished: ", model.evaluate(test_ds.batch(parsed.batch_size)))
    print("Done")
