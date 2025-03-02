import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap

def epsilon_greedy_action(Q, state, current_epsilon, n_actions=4):
    """
    Choose an action from state using an epsilon-greedy policy.
    - With probability current_epsilon, choose a random action.
    - Otherwise, choose the action with the highest Q value.
    """
    if random.random() < current_epsilon:
        return random.randint(0, n_actions - 1)
    else:
        if state not in Q:
            return random.randint(0, n_actions - 1)
        max_q = max(Q[state].values())
        actions = [a for a, q in Q[state].items() if q == max_q]
        return random.choice(actions)

def initialize_Q(env):
    """
    Initialize the Q table: for every state (0 to n_states-1) and action (0,1,2,3), set Q value to 0.
    """
    from collections import defaultdict
    Q = defaultdict(lambda: {a: 0.0 for a in range(4)})
    return Q

def extract_policy(Q, env):
    """
    Extract the policy from the Q table:
    - For the goal and hole states, return None.
    - For other states, return the action that has the highest Q value.
    """
    policy = {}
    for s in range(env.n_states):
        if s == env.goal or s in env.holes:
            policy[s] = None
        else:
            best_action = max(Q[s], key=Q[s].get)
            policy[s] = best_action
    return policy

def print_policy_grid(policy, env):
    """
    Print the policy as a grid:
    - '↑', '→', '↓', '←' represent actions.
    - S: start, G: goal, H: hole.
    """
    arrow_map = {0: '↑', 1: '→', 2: '↓', 3: '←'}
    grid = [['' for _ in range(env.grid_size)] for _ in range(env.grid_size)]
    for s in range(env.n_states):
        row, col = divmod(s, env.grid_size)
        if s == env.start:
            grid[row][col] = 'S'
        elif s == env.goal:
            grid[row][col] = 'G'
        elif s in env.holes:
            grid[row][col] = 'H'
        else:
            action = policy.get(s)
            grid[row][col] = arrow_map[action] if action is not None else '0'
    for row in grid:
        print(' '.join(row))
    print()

def plot_reward_with_moving_average(reward_list, window_size=100):
    """
    Plot the raw reward and its moving average (MA(window_size)).
    """
    rewards_np = np.array(reward_list)
    x = np.arange(rewards_np.shape[0])
    plt.scatter(x, rewards_np, s=1, color='purple', alpha=0.6, label='Raw Reward')
    reward_torch = torch.tensor(rewards_np, dtype=torch.float)
    if reward_torch.shape[0] >= window_size:
        means = reward_torch.unfold(0, window_size, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(window_size - 1), means))
        plt.plot(x, means.numpy(), color='red', label=f'MA({window_size})')
    plt.legend()

def get_training_path(policy, env, max_steps=50):
    """
    Generate a training path from the start state following the policy.
    To avoid infinite loops, a maximum number of steps is enforced.
    """
    state = env.start
    path = [state]
    visited = {state}
    for _ in range(max_steps):
        if state == env.goal or state in env.holes:
            break
        action = policy.get(state)
        if action is None:
            break
        next_state = env._next_state(state, action)
        if next_state in visited:
            break
        path.append(next_state)
        visited.add(next_state)
        state = next_state
    return path

def draw_path_on_map(ax, env, path):
    """
    Draw the Frozen Lake map on the given axes:
    - Draw grid lines and obstacles (semi-transparent black rectangles).
    - Mark the start with "Start" (orange) and goal with "Goal" (purple).
    - Use arrows to indicate the path direction.
    """
    grid_size = env.grid_size
    for x in range(grid_size + 1):
        ax.axhline(x, color='gray', linewidth=1)
        ax.axvline(x, color='gray', linewidth=1)
    for hole in env.holes:
        row, col = divmod(hole, grid_size)
        ax.add_patch(plt.Rectangle((col, grid_size - row - 1), 1, 1, color='black', alpha=0.5))
    start_row, start_col = divmod(env.start, grid_size)
    goal_row, goal_col = divmod(env.goal, grid_size)
    ax.text(start_col + 0.5, grid_size - start_row - 0.5, 'Start', color='orange',
            fontsize=12, ha='center', va='center')
    ax.text(goal_col + 0.5, grid_size - goal_row - 0.5, 'Goal', color='purple',
            fontsize=12, ha='center', va='center')
    path_coords = []
    for state in path:
        row, col = divmod(state, grid_size)
        path_coords.append((col + 0.5, grid_size - row - 0.5))
    for i in range(len(path_coords) - 1):
        start_coord = path_coords[i]
        end_coord = path_coords[i + 1]
        ax.annotate('', xy=end_coord, xytext=start_coord,
                    arrowprops=dict(arrowstyle="->", color='blue', lw=2))
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_title("Training Path")

def plot_heatmap(Q, env, ax):
    """
    Plot a heatmap of action values for each state:
    - Each state is represented by a square divided into four triangles (by its diagonals),
      each corresponding to one of the actions (0: Up, 1: Right, 2: Down, 3: Left).
    - The value is displayed (formatted to two decimals) in each triangle.
    - Color maps from blue (low) to red (high).
    """
    grid_size = env.grid_size
    all_q = [Q[s][a] for s in range(env.n_states) for a in range(4)]
    q_min = min(all_q)
    q_max = max(all_q)
    cmap = LinearSegmentedColormap.from_list("blue_red", ["blue", "red"])
    norm = Normalize(vmin=q_min, vmax=q_max)

    fontsize = 8 if grid_size <= 6 else max(4, 8 - (grid_size - 6))

    for s in range(env.n_states):
        row, col = divmod(s, grid_size)
        bottom = grid_size - row - 1
        left = col
        top = bottom + 1
        right = left + 1
        center = (left + 0.5, bottom + 0.5)
        vertices0 = [(left, top), (right, top), center]
        vertices1 = [(right, top), (right, bottom), center]
        vertices2 = [(right, bottom), (left, bottom), center]
        vertices3 = [(left, bottom), (left, top), center]
        for action, vertices in enumerate([vertices0, vertices1, vertices2, vertices3]):
            q_val = Q[s][action]
            color = cmap(norm(q_val))
            triangle = Polygon(vertices, closed=True, facecolor=color, edgecolor='k')
            ax.add_patch(triangle)
            centroid = np.mean(np.array(vertices), axis=0)
            ax.text(centroid[0], centroid[1], f"{q_val:.2f}", ha='center', va='center',
                    fontsize=fontsize, color='black')
    for i in range(grid_size + 1):
        ax.axhline(i, color='gray', linewidth=1)
        ax.axvline(i, color='gray', linewidth=1)
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_xticks(np.arange(grid_size + 1))
    ax.set_yticks(np.arange(grid_size + 1))
    ax.set_title("Action Value Heatmap")
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
