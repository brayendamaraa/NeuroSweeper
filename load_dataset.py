import h5py
import numpy as np
import tensorflow as tf

def one_hot_state(state):
    # state: (H, W) with values 0..10
    h, w = state.shape
    out = np.zeros((h, w, 11), dtype=np.float32)
    for v in range(11):
        out[:, :, v] = (state == v)
    return out


def load_dataset(h5_path, batch_size=32, shuffle=True):
    f = h5py.File(h5_path, "r")
    states = f["states"]
    actions = f["actions"]

    n = states.shape[0]

    def generator():
        idxs = np.arange(n)
        if shuffle:
            np.random.shuffle(idxs)

        for i in idxs:
            s = one_hot_state(states[i])
            a = actions[i].astype(np.float32)
            yield s, a[..., None]  # (H, W, 1)

    # infer shape
    H, W = states.shape[1], states.shape[2]

    return tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(H, W, 11), dtype=tf.float32),
            tf.TensorSpec(shape=(H, W, 1), dtype=tf.float32),
        )
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
