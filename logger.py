import h5py
import numpy as np

class H5Logger:
    def __init__(self, path, board_shape):
        self.path = path
        self.board_shape = board_shape
        self._open()

    def _open(self):
        self.f = h5py.File(self.path, "a")

        if "states" not in self.f:
            self.f.create_dataset(
                "states",
                shape=(0, *self.board_shape),
                maxshape=(None, *self.board_shape),
                dtype=np.int8,
                chunks=True,
            )

            self.f.create_dataset(
                "actions",
                shape=(0, *self.board_shape),
                maxshape=(None, *self.board_shape),
                dtype=np.int8,
                chunks=True,
            )

    def append(self, state, action):
        states = self.f["states"]
        actions = self.f["actions"]

        n = states.shape[0]

        states.resize(n + 1, axis=0)
        actions.resize(n + 1, axis=0)

        states[n] = state
        actions[n] = action

    def close(self):
        self.f.close()
