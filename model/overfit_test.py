import random, os
import numpy as np
import tensorflow as tf
from train_test import get_vector_files_and_labels, make_dataset, build_model, CLASSES, TARGET_SIZE

# pick a very small subset to overfit
paths, labels = get_vector_files_and_labels(CLASSES)
if len(paths) < 8:
    raise SystemExit("Not enough files for overfit test")

# sample 8 examples (keep labels as-is)
sel_idx = random.sample(range(len(paths)), 8)
sel_paths = [paths[i] for i in sel_idx]
sel_labels = [labels[i] for i in sel_idx]

ds = make_dataset(sel_paths, sel_labels, batch_size=4, shuffle=True)
model = build_model(input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3), num_classes=len(CLASSES))
model.fit(ds, epochs=100, verbose=2)
# expect train loss to go down toward 0 and acc -> 1.0