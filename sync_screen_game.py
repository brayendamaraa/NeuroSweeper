import pyautogui
import numpy as np
import tensorflow as tf
from model import wmse


expert = {
            "rows": 16,
            "cols": 30,
            "mines": 99,
            "sad_x": 750,
            "sad_y": 176,
        }
intermediate = {
            "rows": 16,
            "cols": 16,
            "mines": 40,
            "sad_x": 525,
            "sad_y": 175,
        }
beginner = {
            "rows": 9,
            "cols": 9,
            "mines": 10,
            "sad_x": 414,
            "sad_y": 169,
        }

TOP_LEFT_X = 262
TOP_LEFT_Y = 219
CELL_SIZE = 32

COLOR_MAP = {
    (0, 0, 255): 1, (0, 128, 0): 2, (255, 0, 0): 3,
    (0, 0, 128): 4, (128, 0, 0): 5, (0, 128, 128): 6,
    (0, 0, 0): 7, (128, 128, 128): 8, (198, 198, 198): 0
}

class Cell:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.x = TOP_LEFT_X + col * CELL_SIZE + CELL_SIZE // 2
        self.y = TOP_LEFT_Y + row * CELL_SIZE + CELL_SIZE // 2
        self.state = "unrevealed"  # "unrevealed", "flagged", "revealed"
        self.value = None  # 0-8 for revealed cells, "F" for flagged, None for unrevealed

class Board:
    def __init__(self, rows=None, cols=None, mines=None, difficulty=None, seed=None, sad_x=None, sad_y=None):
        self.seed = seed
        if difficulty is not None:
            if difficulty == "expert":
                self.set_settings(**expert)
            elif difficulty == "intermediate":
                self.set_settings(**intermediate)
            elif difficulty == "beginner":
                self.set_settings(**beginner)
        else:
            self.set_settings(rows, cols, mines, sad_x, sad_y)
        self.grid = [[Cell(r, c) for c in range(self.cols)] for r in range(self.rows)]
        
    def set_settings(self, rows, cols, mines, sad_x, sad_y):
        if mines >= rows * cols:
            raise ValueError("Too many mines.")
        self.rows = rows
        self.cols = cols
        self.num_mines = mines
        self.sad_x = sad_x
        self.sad_y = sad_y
        

    def check_game_over(self):

        # 1. LOGIKA KALAH: Mencocokkan warna pixel
        if pyautogui.pixelMatchesColor(self.sad_x, self.sad_y, (0, 0, 0), tolerance=10):
            print("Game Over! Kamu Kalah.")
            return True

        # 2. LOGIKA MENANG: Cek jumlah sel yang masih bernilai 9 (unrevealed)
        if len(self.unrevealed_cells) == 0:
            print("Selamat! Kamu Menang.")
            return True
            
        return False

    def sync_with_screen(self):
        screenshot = pyautogui.screenshot()
        flags_count = 0
        for r in range(self.rows):
            for c in range(self.cols):
                cell = self.grid[r][c]
                rgb_center = screenshot.getpixel((cell.x, cell.y))
                if self.is_cell_revealed(screenshot, cell.x, cell.y):
                    cell.state = "revealed"
                    found_val = 9
                    min_dist = 999
                    for color, val in COLOR_MAP.items():
                        dist = np.linalg.norm(np.array(rgb_center) - np.array(color))
                        if dist < 25 and dist < min_dist:
                            min_dist = dist
                            found_val = val
                    cell.value = found_val
                else:
                    dist_hitam = np.linalg.norm(np.array(rgb_center) - np.array((0, 0, 0)))
                    dist_merah = np.linalg.norm(np.array(rgb_center) - np.array((255, 0, 0)))
                    if dist_hitam < 30 or dist_merah < 50:
                        cell.state = "flagged"
                        cell.value = "10"
                        flags_count += 1
                    else:
                        cell.state = "unrevealed"
                        cell.value = 9
        self.total_flags = flags_count
        self.mines_remaining = self.num_mines - flags_count
        self.unrevealed_cells = [cell for row in self.grid for cell in row if cell.state == "unrevealed"]

    def is_cell_revealed(self, screen, x, y):
        edge_pixel = screen.getpixel((x - 14, y - 14)) 
        return not (edge_pixel[0] > 220 and edge_pixel[1] > 220)
    
    def get_board_state(self):
        matrix = np.zeros((self.rows, self.cols), dtype=int)
        for r in range(self.rows):
            for c in range(self.cols):
                val = self.grid[r][c].value
                matrix[r, c] = val
        return matrix
    
    def make_action(self, action):
        if action is None:
            return
        kind, r, c = action
        cell = self.grid[r][c]
        if kind == 'flag' and cell.state == "unrevealed":
            pyautogui.rightClick(cell.x, cell.y)
        elif kind == 'reveal' and cell.state == "unrevealed":
            pyautogui.leftClick(cell.x, cell.y)