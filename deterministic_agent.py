from itertools import product, combinations
import random
from game import Minesweeper

class MinesweeperAgent:
    def __init__(self, game):
        self.game = game

    def _neighbors(self, r, c):
        return self.game._neighbors(r, c)

    def simple_rule_action(self):
        board = self.game.get_board_state()

        for r in range(self.game.rows):
            for c in range(self.game.cols):
                cell = board[r][c]

                # only numbered cells
                if not (0 <= cell <= 8):
                    continue

                unrevealed = []
                flagged = 0

                for nr, nc in self._neighbors(r, c):
                    if self.game.flagged[nr][nc]:
                        flagged += 1
                    elif not self.game.revealed[nr][nc]:
                        unrevealed.append((nr, nc))

                if not unrevealed:
                    continue

                # RULE 1: all unrevealed are mines → FLAG
                if cell == flagged + len(unrevealed):
                    ur, uc = unrevealed[0]
                    return ('flag', ur, uc)

                # RULE 2: all unrevealed are safe → REVEAL
                if cell == flagged:
                    ur, uc = unrevealed[0]
                    return ('reveal', ur, uc)

        return None


    def solve_linked_pairs_action(self):
        board = self.game.get_board_state()
        rows, cols = self.game.rows, self.game.cols

        # -------------------------------------------------
        # 1. Collect frontier NUMBER constraints
        #    Each constraint: (remaining_mines, unrevealed_neighbors)
        # -------------------------------------------------
        constraints = []

        for r in range(rows):
            for c in range(cols):
                v = board[r][c]
                if not (0 <= v <= 8):
                    continue

                unrevealed = set()
                flagged = 0

                for nr, nc in self._neighbors(r, c):
                    if self.game.flagged[nr][nc]:
                        flagged += 1
                    elif not self.game.revealed[nr][nc]:
                        unrevealed.add((nr, nc))

                if unrevealed:
                    constraints.append((v - flagged, unrevealed))

        # -------------------------------------------------
        # 2. Scan every PAIR of constraints
        # -------------------------------------------------
        for i in range(len(constraints)):
            k1, U1 = constraints[i]

            for j in range(i + 1, len(constraints)):
                k2, U2 = constraints[j]

                # must share at least one unrevealed cell
                if not (U1 & U2):
                    continue

                frontier = list(U1 | U2)

                # hard cutoff for performance
                if len(frontier) > 10:
                    continue

                index = {cell: idx for idx, cell in enumerate(frontier)}
                eq1 = [index[c] for c in U1]
                eq2 = [index[c] for c in U2]

                # -------------------------------------------------
                # 3. Enumerate valid mine configurations
                # -------------------------------------------------
                valid_configs = []

                for cfg in product((0, 1), repeat=len(frontier)):
                    if sum(cfg[i] for i in eq1) != k1:
                        continue
                    if sum(cfg[i] for i in eq2) != k2:
                        continue
                    valid_configs.append(cfg)

                if not valid_configs:
                    continue

                # -------------------------------------------------
                # 4. Deduce forced action
                # -------------------------------------------------
                for idx, (r0, c0) in enumerate(frontier):
                    if self.game.flagged[r0][c0] or self.game.revealed[r0][c0]:
                        continue

                    values = {cfg[idx] for cfg in valid_configs}

                    # always a mine
                    if values == {1}:
                        return ('flag', r0, c0)

                    # always safe
                    if values == {0}:
                        return ('reveal', r0, c0)

        return None
    
    def solve_triple_linked_pairs_action(self):
        board = self.game.get_board_state()
        rows, cols = self.game.rows, self.game.cols

        # -------------------------------------------------
        # 1. Collect frontier NUMBER constraints
        # -------------------------------------------------
        constraints = []

        for r in range(rows):
            for c in range(cols):
                v = board[r][c]
                if not (0 <= v <= 8):
                    continue

                unrevealed = set()
                flagged = 0

                for nr, nc in self._neighbors(r, c):
                    if self.game.flagged[nr][nc]:
                        flagged += 1
                    elif not self.game.revealed[nr][nc]:
                        unrevealed.add((nr, nc))

                if unrevealed:
                    constraints.append((v - flagged, unrevealed))

        n = len(constraints)

        # -------------------------------------------------
        # 2. Choose center constraint A
        # -------------------------------------------------
        for a in range(n):
            kA, UA = constraints[a]

            linked = [
                b for b in range(n)
                if b != a and UA & constraints[b][1]
            ]

            if len(linked) < 2:
                continue

            # -------------------------------------------------
            # 3. Try all (B, C) pairs
            # -------------------------------------------------
            for i in range(len(linked)):
                b = linked[i]
                kB, UB = constraints[b]

                for j in range(i + 1, len(linked)):
                    c = linked[j]
                    kC, UC = constraints[c]

                    frontier = list(UA | UB | UC)

                    # performance cutoff
                    if len(frontier) > 12:
                        continue

                    index = {cell: idx for idx, cell in enumerate(frontier)}

                    eqA = [index[x] for x in UA]
                    eqB = [index[x] for x in UB]
                    eqC = [index[x] for x in UC]

                    # -------------------------------------------------
                    # 4. Enumerate valid configs
                    # -------------------------------------------------
                    valid_configs = []

                    for cfg in product((0, 1), repeat=len(frontier)):
                        if sum(cfg[i] for i in eqA) != kA:
                            continue
                        if sum(cfg[i] for i in eqB) != kB:
                            continue
                        if sum(cfg[i] for i in eqC) != kC:
                            continue
                        valid_configs.append(cfg)

                    if not valid_configs:
                        continue

                    # -------------------------------------------------
                    # 5. Deduce forced action
                    # -------------------------------------------------
                    for idx, (r0, c0) in enumerate(frontier):
                        if self.game.flagged[r0][c0] or self.game.revealed[r0][c0]:
                            continue

                        values = {cfg[idx] for cfg in valid_configs}

                        if values == {1}:
                            return ('flag', r0, c0)

                        if values == {0}:
                            return ('reveal', r0, c0)

        return None
    
    def solve_endgame_remaining_mines_action(self):
        board = self.game.get_board_state()
        rows, cols = self.game.rows, self.game.cols

        # -------------------------------------------------
        # 0. Count remaining mines
        # -------------------------------------------------
        flagged_count = sum(
            self.game.flagged[r][c]
            for r in range(rows)
            for c in range(cols)
        )
        remaining_mines = self.game.num_mines - flagged_count

        # Hard cutoff
        if remaining_mines <= 0 or remaining_mines > 10:
            return None

        # -------------------------------------------------
        # 1. Collect unknown cells
        # -------------------------------------------------
        unknown = [
            (r, c)
            for r in range(rows)
            for c in range(cols)
            if not self.game.revealed[r][c]
            and not self.game.flagged[r][c]
        ]

        # Too many combinations
        if len(unknown) > 15:
            return None

        # -------------------------------------------------
        # 2. Collect numbered constraints
        # -------------------------------------------------
        constraints = []

        for r in range(rows):
            for c in range(cols):
                v = board[r][c]
                if not (0 <= v <= 8):
                    continue

                unrevealed = []
                flagged_here = 0

                for nr, nc in self._neighbors(r, c):
                    if self.game.flagged[nr][nc]:
                        flagged_here += 1
                    elif not self.game.revealed[nr][nc]:
                        unrevealed.append((nr, nc))

                if unrevealed:
                    constraints.append((v - flagged_here, unrevealed))

        if not constraints:
            return None

        # -------------------------------------------------
        # 3. Enumerate valid mine configurations
        # -------------------------------------------------
        valid_configs = []

        for mines in combinations(unknown, remaining_mines):
            mine_set = set(mines)
            valid = True

            for k, cells in constraints:
                if sum(c in mine_set for c in cells) != k:
                    valid = False
                    break

            if valid:
                valid_configs.append(mine_set)

        if not valid_configs:
            return None

        # -------------------------------------------------
        # 4. Deduce forced actions
        # -------------------------------------------------
        for r, c in unknown:
            states = {(r, c) in cfg for cfg in valid_configs}

            if states == {True}:
                return ('flag', r, c)

            if states == {False}:
                return ('reveal', r, c)

        return None
    
    def random_reveal_cell(self):
        unrevealed = self.game.get_unrevealed_cells()
        if not unrevealed:
            return None
        return random.choice(unrevealed)
    
    def next_action(self):
        """
        Decide exactly ONE action to take.
        Returns:
            ('flag', r, c)
            ('reveal', r, c)
            or None if no move possible
        """

        # Deterministic solvers (order matters, but no priority logic inside them)
        solvers = [
            self.simple_rule_action,
            self.solve_linked_pairs_action,
            self.solve_triple_linked_pairs_action,
            self.solve_endgame_remaining_mines_action,
        ]

        for solver in solvers:
            action = solver()
            if action is not None:
                return action

        # Fallback: random reveal (forced guess)
        cell = self.random_reveal_cell()
        if cell is not None:
            r, c = cell
            return ('reveal', r, c)

        return None


if __name__ == "__main__":
    def is_win(game):
        for r in range(game.rows):
            for c in range(game.cols):
                if (r, c) not in game.mines and not game.revealed[r][c]:
                    return False
        return True
    
    num_games = 1
    win = 0
    for i in range(num_games):
        game = Minesweeper(difficulty="expert", seed=None)
        agent = MinesweeperAgent(game)

        # safe first move
        game.make_move(game.rows // 2, game.cols // 2)
        while not game.game_over:
            action = agent.next_action()
            if action is None:
                break

            kind, r, c = action
            if kind == 'flag':
                game.flag_cell(r, c)
            elif kind == 'reveal':
                game.make_move(r, c)
            state = game.get_board_state()
            for row in state:
                print(' '.join(f"{v:2}" for v in row))
            print()
            
        if is_win(game):
            win += 1
    
    print(f"Win rate: {win}/{num_games} = {win / num_games:.2%}")