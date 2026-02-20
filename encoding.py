import numpy as np

def encode_action(game, action):
    """
    Encode a single agent action into an (H, W) matrix.

    action:
        ('flag', r, c) or ('reveal', r, c)

    Returns:
        mat[r, c] = -1  for flag
        mat[r, c] = +1  for reveal
        mat = 0 elsewhere
    """
    h, w = game.rows, game.cols
    mat = np.zeros((h, w), dtype=np.int8)

    if action is None:
        return mat  # no-op (should rarely happen)

    kind, r, c = action

    if kind == 'flag':
        mat[r, c] = -1
    elif kind == 'reveal':
        mat[r, c] = 1
    else:
        raise ValueError(f"Unknown action type: {kind}")

    return mat