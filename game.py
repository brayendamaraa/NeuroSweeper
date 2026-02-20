import random
from collections import deque

expert = {"rows": 16, "cols": 30, "mines": 99}
intermediate = {"rows": 16, "cols": 16, "mines": 40}
beginner = {"rows": 9, "cols": 9, "mines": 10}

class Minesweeper:
    UNREVEALED = 9
    MINE = 10

    def __init__(self, rows=None, cols=None, mines=None, difficulty=None, seed=None):
        self.seed = seed
        if difficulty is not None:
            if difficulty == "expert":
                self.set_settings(**expert)
            elif difficulty == "intermediate":
                self.set_settings(**intermediate)
            elif difficulty == "beginner":
                self.set_settings(**beginner)
        else:
            self.set_settings(rows, cols, mines)

    def set_settings(self, rows, cols, mines):
        if mines >= rows * cols:
            raise ValueError("Too many mines.")
        self.rows = rows
        self.cols = cols
        self.num_mines = mines
        self.reset()

    def reset(self):
        if self.seed is not None:
            random.seed(self.seed)
        self.mines = set()
        self.revealed = [[False]*self.cols for _ in range(self.rows)]
        self.flagged = [[False]*self.cols for _ in range(self.rows)]
        self.game_over = False
        self.first_move = True

    def _neighbors(self, r, c):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    yield nr, nc

    def _neighbor_mine_count(self, r, c):
        return sum((nr, nc) in self.mines for nr, nc in self._neighbors(r, c))

    def _place_mines_safe(self, r, c):
        forbidden = {(r, c)} | set(self._neighbors(r, c))
        cells = [
            (i, j)
            for i in range(self.rows)
            for j in range(self.cols)
            if (i, j) not in forbidden
        ]
        self.mines = set(random.sample(cells, self.num_mines))

    def make_move(self, r, c):
        if self.game_over or self.flagged[r][c]:
            return "invalid"

        if self.first_move:
            self._place_mines_safe(r, c)
            self.first_move = False

        if (r, c) in self.mines:
            self.revealed[r][c] = True
            self.game_over = True
            return "mine"

        count = self._neighbor_mine_count(r, c)
        self.revealed[r][c] = True

        if count == 0:
            queue = deque([(r, c)])
            while queue:
                cr, cc = queue.popleft()
                for nr, nc in self._neighbors(cr, cc):
                    if not self.revealed[nr][nc] and not self.flagged[nr][nc]:
                        self.revealed[nr][nc] = True
                        if self._neighbor_mine_count(nr, nc) == 0:
                            queue.append((nr, nc))
            return "zero"

        return "safe"


    def flag_cell(self, r, c):
        if not self.revealed[r][c]:
            self.flagged[r][c] = True

    def get_board_state(self):
        board = [[self.UNREVEALED]*self.cols for _ in range(self.rows)]
        for r in range(self.rows):
            for c in range(self.cols):
                if self.flagged[r][c]:
                    board[r][c] = 10
                elif self.revealed[r][c]:
                    board[r][c] = self._neighbor_mine_count(r, c)
        return board
    
    def get_unrevealed_cells(self):
        return [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if not self.revealed[r][c] and not self.flagged[r][c]
        ]
