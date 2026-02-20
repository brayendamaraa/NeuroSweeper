import time
import numpy as np
import pyautogui
import keyboard
import sys
import os
from neurosweeper import NeuroSweeper
from sync_screen_game import Board

# Global stop flag
STOP_REQUESTED = False


def stop_program():
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n[!] STOP REQUESTED - Finishing current step...")


def play_games(model_path, difficulty="intermediate", max_games=100):

    global STOP_REQUESTED

    wins = 0
    losses = 0

    # Register ESC key as emergency stop
    keyboard.add_hotkey("esc", stop_program)

    print("Press ESC anytime to stop safely.\n")

    for game_number in range(max_games):

        if STOP_REQUESTED:
            break

        board = Board(difficulty=difficulty)
        agent = NeuroSweeper(board, model_path)

        # Reset board
        pyautogui.click(board.sad_x, board.sad_y)

        # First move (center)
        r = board.rows // 2
        c = board.cols // 2
        first_cell = board.grid[r][c]
        pyautogui.leftClick(first_cell.x, first_cell.y)

        while True:
            if STOP_REQUESTED:
                break

            action = agent.predict_action()

            if action is None:
                board.sync_with_screen()
                if len(board.unrevealed_cells) == 0:
                    break
                cell = np.random.choice(board.unrevealed_cells)
                action = ('reveal', cell.row, cell.col)

            board.make_action(action)

            board.sync_with_screen()

            if board.check_game_over():
                if len(board.unrevealed_cells) == 0:
                    wins += 1
                else:
                    losses += 1
                break

    keyboard.unhook_all_hotkeys()
    sys.exit(0)


if __name__ == "__main__":
    model_path = os.path.join(os.path.dirname(__file__), 'neurosweeper_weight.h5')
    play_games(model_path, difficulty="expert", max_games=1)