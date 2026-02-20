from game import Minesweeper
from deterministic_agent import MinesweeperAgent
from logger import H5Logger
from encoding import encode_action
import numpy as np


def run_games(n_games, difficulty, out_file):
    game = Minesweeper(difficulty=difficulty)
    agent = MinesweeperAgent(game)

    logger = H5Logger(out_file, (game.rows, game.cols))

    for g in range(n_games):
        game.reset()

        # First move (safe random reveal)
        r = np.random.randint(game.rows)
        c = np.random.randint(game.cols)
        game.make_move(r, c)

        while not game.game_over:

            # Ask agent for ONE action
            action = agent.next_action()

            if action is None:
                break  # no legal action

            # Encode BEFORE applying action
            state = np.array(game.get_board_state(), dtype=np.int8)
            action_mat = encode_action(game, action)

            logger.append(state, action_mat)

            # Apply action
            kind, r, c = action
            if kind == 'flag':
                game.flag_cell(r, c)
            elif kind == 'reveal':
                game.make_move(r, c)
            else:
                raise ValueError(kind)

        if (g + 1) % 100 == 0:
            print(f"{g + 1} games collected")

    logger.close()


if __name__ == "__main__":
    run_games(
        n_games=1000,
        difficulty="expert",
        out_file="minesweeper_expert_action.h5"
    )