import numpy as np
from tensorflow.keras.models import load_model
from model import wmse

class NeuroSweeper:
    def __init__(self, board, model_path, threshold=0.05):
        self.board = board
        self.model = load_model(model_path, custom_objects={'wmse': wmse})
        self.threshold = threshold  # ignore very small outputs

    def one_hot_encode(self, matrix):
        rows, cols = matrix.shape
        one_hot = np.zeros((rows, cols, 11), dtype=np.float32)

        for r in range(rows):
            for c in range(cols):
                val = matrix[r, c]
                one_hot[r, c, val] = 1.0

        return np.expand_dims(one_hot, axis=0)

    def predict_action(self):
        self.board.sync_with_screen()
        state_matrix = self.board.get_board_state()
        input_tensor = self.one_hot_encode(state_matrix)

        # Predict
        output = self.model.predict(input_tensor, verbose=0)

        # Shape: (1, rows, cols, 1)
        action_matrix = output[0, :, :, 0]

        rows, cols = action_matrix.shape

        best_value = 0
        best_action = None

        for r in range(rows):
            for c in range(cols):
                cell = self.board.grid[r][c]

                if cell.state != "unrevealed":
                    continue

                value = action_matrix[r, c]

                # Ignore weak confidence
                if abs(value) < self.threshold:
                    continue

                if abs(value) > abs(best_value):
                    best_value = value
                    best_action = (r, c)

        if best_action is None:
            return None

        r, c = best_action

        if best_value > 0:
            return ('reveal', r, c)
        else:
            return ('flag', r, c)

    def step(self):
        action = self.predict_action()
        self.board.make_action(action)