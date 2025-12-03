import os
import glob
import json
import numpy as np
from typing import List, Tuple

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# CONFIG
BASE_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
VECTOR_SUFFIX = "_vectorized"
TARGET_SIZE = (128, 128)     # resize all images to this
BATCH_SIZE = 5
AUTOTUNE = tf.data.AUTOTUNE
EPOCHS = 30
MODEL_OUT = os.path.join(os.path.dirname(__file__), '..', 'model', 'astro_cnn')

# CLASSES: adjust if you have other folders
CLASSES = ["asteroids", "black_holes", "planets", "stars"]


def get_vector_files_and_labels(classes: List[str], base_dir: str = BASE_DATA_DIR) -> Tuple[List[str], List[int]]:
    paths = []
    labels = []
    for idx, cls in enumerate(classes):
        vec_dir = os.path.join(base_dir, f"{cls}{VECTOR_SUFFIX}")
        if not os.path.isdir(vec_dir):
            print(f"Warning: directory not found: {vec_dir}")
            continue
        # take only .npy files
        files = sorted(glob.glob(os.path.join(vec_dir, "*.npy")))
        for f in files:
            paths.append(os.path.abspath(f))
            labels.append(idx)
    return paths, labels


def _load_npy(path):
    # this runs in numpy, returns float32 array scaled 0..1
    arr = np.load(path.decode("utf-8"))
    # If image has integer dtype, convert; otherwise cast
    arr = arr.astype(np.float32)
    # if arr values appear >1 assume 0-255
    if arr.max() > 1.1:
        arr = arr / 255.0
    # ensure channel dimension
    if arr.ndim == 2:
        arr = np.expand_dims(arr, -1)
    return arr


def make_dataset(file_paths: List[str], labels: List[int], batch_size: int = BATCH_SIZE, shuffle: bool = True):
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    def _read(path, label):
        img = tf.numpy_function(_load_npy, [path], tf.float32)
        img.set_shape([None, None, None])  # unknown dims until resized
        # ensure 3 channels for model (RGB). If single-channel, tile to 3.
        def fix_channels(x):
            if x.shape[-1] == 1:
                return np.concatenate([x, x, x], axis=-1)
            elif x.shape[-1] == 4:
                # drop alpha if present
                return x[..., :3]
            return x
        img = tf.numpy_function(lambda x: fix_channels(x), [img], tf.float32)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, TARGET_SIZE)
        # clip to [0,1]
        img = tf.clip_by_value(img, 0.0, 1.0)
        return img, label

    ds = ds.map(_read, num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=1024)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def build_model(input_shape=(128, 128, 3), num_classes=4):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs, name="astro_cnn")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), # LEARNING RATE
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# replace or add this function and call it instead of build_model when experimenting
def build_model_tl(input_shape=(128,128,3), num_classes=4):
    base = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base.trainable = False  # freeze for initial training
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs * 255.0  # base expects pixels in 0..255 before preprocess
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    # prepare dirs, seeds and environment info
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    tf.random.set_seed(42)
    np.random.seed(42)
    print("tf version:", tf.__version__, "numpy:", np.__version__)
    try:
        gpus = tf.config.list_physical_devices('GPU')
        print("GPUs:", gpus)
    except Exception:
        print("Could not enumerate GPUs")

    file_paths, labels = get_vector_files_and_labels(CLASSES)
    if not file_paths:
        raise RuntimeError("No .npy files found. Run vectorize.py first or point BASE_DATA_DIR correctly.")

    # split lists into train/val/test
    fp_train, fp_temp, y_train, y_temp = train_test_split(file_paths, labels, test_size=0.25, stratify=labels, random_state=42)
    fp_val, fp_test, y_val, y_test = train_test_split(fp_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    train_ds = make_dataset(fp_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
    val_ds = make_dataset(fp_val, y_val, batch_size=BATCH_SIZE, shuffle=False)
    test_ds = make_dataset(fp_test, y_test, batch_size=BATCH_SIZE, shuffle=False)

    # Debugging: dataset / input checks
    print("TOTAL samples:", len(file_paths))
    counts = {cls: sum(1 for l in labels if l == i) for i, cls in enumerate(CLASSES)}
    print("Per-class counts (from labels):", counts)
    print(f"Split sizes -> train: {len(fp_train)}  val: {len(fp_val)}  test: {len(fp_test)}")
    print(f"Batch sizes -> train_batches: {len(train_ds)}  val_batches: {len(val_ds)}  test_batches: {len(test_ds)}")
    # inspect one batch
    try:
        batch = next(iter(train_ds))
        imgs, labs = batch
        print("Sample batch -> imgs.shape:", imgs.shape, "dtype:", imgs.dtype)
        print("Sample batch -> min/max:", float(tf.reduce_min(imgs).numpy()), float(tf.reduce_max(imgs).numpy()))
        print("Sample batch -> mean/std:", float(tf.reduce_mean(imgs).numpy()), float(tf.math.reduce_std(imgs).numpy()))
        print("Sample batch labels unique:", np.unique(labs.numpy(), return_counts=True))
    except Exception as e:
        print("Warning: couldn't inspect a training batch:", e)

    # Transfer learning: short frozen training then fine-tune
    model = build_model_tl(input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3), num_classes=len(CLASSES))
    model.summary()

    initial_epochs = min(8, EPOCHS)
    fine_tune_epochs = max(0, EPOCHS - initial_epochs)

    ckpt = tf.keras.callbacks.ModelCheckpoint(MODEL_OUT + ".keras", save_best_only=True, monitor="val_accuracy", mode="max")
    early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)

    print(f"Training head for {initial_epochs} epochs (base frozen)...")
    model.fit(train_ds, validation_data=val_ds, epochs=initial_epochs, callbacks=[ckpt, early])

    if fine_tune_epochs > 0:
        # unfreeze last blocks for fine-tuning
        base = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model) and 'mobilenet' in layer.name.lower():
                base = layer
                break
        if base is None:
            # fallback: unfreeze all
            model.trainable = True
        else:
            # unfreeze last ~20 layers of base (adjust as needed)
            base.trainable = True
            for l in base.layers[:-20]:
                l.trainable = False

        # recompile with lower LR for fine-tuning
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print(f"Fine-tuning for {fine_tune_epochs} epochs (unfrozen) ...")
        model.fit(train_ds, validation_data=val_ds, epochs=fine_tune_epochs, callbacks=[ckpt, early])

    # final evaluate and save
    results = model.evaluate(test_ds)
    print("Test loss, Test accuracy:", results)

    # predictions + classification report
    y_true = []
    y_pred = []
    for batch_x, batch_y in test_ds:
        preds = model.predict(batch_x)
        y_true.extend(batch_y.numpy().tolist())
        y_pred.extend(np.argmax(preds, axis=1).tolist())
    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4))

    # ensure final model and label map saved
    final_path = MODEL_OUT + ".keras"
    model.save(final_path, include_optimizer=False)
    with open(os.path.join(os.path.dirname(MODEL_OUT), "label_map.json"), "w") as f:
        json.dump({"classes": CLASSES}, f)
    print(f"Saved final model to {final_path}")


if __name__ == "__main__":
    main()