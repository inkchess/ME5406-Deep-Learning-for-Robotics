import numpy as np
import random
from collections import deque


class FrozenLakeEnv:
    """
    Frozen Lake Environment:
    - The grid is of size grid_size x grid_size.
    - The start state is the top-left (state 0) and the goal is the bottom-right (state grid_size*grid_size - 1).
    - Some states are holes; entering a hole gives a reward of -1 and ends the episode.
    - Reaching the goal gives a reward of +1 and ends the episode; all other states have a reward of 0.
    - If explore_bonus_value (non-zero) is set, non-terminal states will yield that bonus (typically a small negative value) to encourage faster episode termination.
    - If custom_holes is provided, it will use the specified obstacles; otherwise, obstacles are randomly generated.
    """

    def __init__(self, grid_size=4, hole_ratio=0.25, seed=None, custom_holes=None, explore_bonus_value=0.0):
        self.grid_size = grid_size
        self.n_states = grid_size * grid_size
        self.start = 0
        self.goal = self.n_states - 1
        self.hole_ratio = hole_ratio
        self.state = self.start
        self.explore_bonus_value = explore_bonus_value  # Exploration bonus value

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if custom_holes is not None:
            # For custom maps, ensure start and goal are not obstacles.
            self.holes = set(custom_holes) - {self.start, self.goal}
        else:
            self._generate_holes()

    def _generate_holes(self):
        num_holes = int(self.n_states * self.hole_ratio)
        valid = False
        while not valid:
            hole_candidates = list(range(1, self.n_states - 1))
            random.shuffle(hole_candidates)
            self.holes = set(hole_candidates[:num_holes])
            if self._check_connectivity():
                valid = True

    def _check_connectivity(self):
        visited = set()
        queue = deque()
        queue.append(self.start)
        visited.add(self.start)
        while queue:
            current = queue.popleft()
            if current == self.goal:
                return True
            for action in [0, 1, 2, 3]:
                next_state = self._next_state(current, action)
                if next_state not in visited and next_state not in self.holes:
                    visited.add(next_state)
                    queue.append(next_state)
        return False

    def _next_state(self, state, action):
        """
        Compute the next state given current state and action.
        Actions:
            0: Up, 1: Right, 2: Down, 3: Left.
        If the movement goes out of bounds, the state remains unchanged.
        """
        row, col = divmod(state, self.grid_size)
        action_map = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        delta = action_map[action]
        new_row = row + delta[0]
        new_col = col + delta[1]
        if new_row < 0 or new_row >= self.grid_size or new_col < 0 or new_col >= self.grid_size:
            return state
        return new_row * self.grid_size + new_col

    def reset(self):
        """Reset the environment and return the initial state."""
        self.state = self.start
        return self.state

    def step(self, action):
        """
        Update the environment state based on the action.
        Returns: (new_state, reward, done, valid_move)
         - If the action moves out-of-bound, the move is considered invalid:
             * The state remains unchanged.
             * reward is 0.
             * done is False.
             * valid_move is False (i.e., this move is not counted in training metrics).
         - If the new state is a hole, reward = -1 and episode ends.
         - If the new state is the goal, reward = +1 and episode ends.
         - Otherwise, reward = 0; if explore_bonus_value is set, then that value is used.
        """
        # Determine current row and col
        row, col = divmod(self.state, self.grid_size)
        action_map = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        delta = action_map[action]
        new_row = row + delta[0]
        new_col = col + delta[1]

        # Check if move is within boundaries
        if new_row < 0 or new_row >= self.grid_size or new_col < 0 or new_col >= self.grid_size:
            # Invalid move: do not count in training metrics
            return self.state, 0, False, False

        # Compute next state for a valid move
        next_state = new_row * self.grid_size + new_col

        if next_state in self.holes:
            reward = -1
            done = True
        elif next_state == self.goal:
            reward = 1
            done = True
        else:
            reward = 0
            done = False
            if self.explore_bonus_value != 0:
                reward = self.explore_bonus_value
        self.state = next_state
        return next_state, reward, done, True

    def render(self, policy=None):
        """
        Print the environment layout.
        - 'S' indicates start, 'G' indicates goal, 'H' indicates hole.
        - Other states display arrows if a policy is provided.
        """
        grid = [['' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        arrow_map = {0: '↑', 1: '→', 2: '↓', 3: '←'}
        for s in range(self.n_states):
            row, col = divmod(s, self.grid_size)
            if s == self.start:
                grid[row][col] = 'S'
            elif s == self.goal:
                grid[row][col] = 'G'
            elif s in self.holes:
                grid[row][col] = 'H'
            else:
                if policy is not None and s in policy:
                    grid[row][col] = arrow_map.get(policy[s], '0')
                else:
                    grid[row][col] = '0'
        for row in grid:
            print(' '.join(row))
        print()
