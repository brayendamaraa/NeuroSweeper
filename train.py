import os
import tensorflow as tf
from load_dataset import load_dataset
from model import build_model, wmse
import h5py


DATA_PATH = os.path.join(os.path.dirname(__file__), 'minesweeper_expert_action.h5')

BATCH_SIZE = 32
dataset = load_dataset(DATA_PATH, batch_size=BATCH_SIZE)

with h5py.File(DATA_PATH, "r") as f:
    total_samples = f["states"].shape[0]


steps_per_epoch = 200
EPOCHS = 100


model = build_model()

model.compile(
    optimizer='adam',
    loss=wmse, metrics=['mse']
)

model.summary()

history = model.fit(
    dataset,
    epochs=EPOCHS,
    batch_size = BATCH_SIZE,
    steps_per_epoch=steps_per_epoch,
)

model.save("neurosweeper_weight.h5")