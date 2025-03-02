import argparse
import os
import time
import textwrap
import random
import numpy as np
import matplotlib.pyplot as plt

from env import FrozenLakeEnv
from utils import (print_policy_grid, plot_reward_with_moving_average, get_training_path,
                   draw_path_on_map, plot_heatmap, extract_policy)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Solve the Frozen Lake problem with RL algorithms.")
parser.add_argument("--method", type=str, default="sarsa", choices=["sarsa", "monte_carlo", "q_learning"],
                    help="RL algorithm to use: sarsa, monte_carlo, q_learning")
parser.add_argument("--map_size", type=int, default=4, help="Grid size (e.g., 4 or 10)")
parser.add_argument("--epsilon", type=float, default=0.1, help="Initial epsilon value")
parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor gamma")
parser.add_argument("--episodes", type=int, default=5000, help="Number of training episodes")
parser.add_argument("--epsilon_decay", type=float, default=0.99, help="Epsilon decay rate per episode")
parser.add_argument("--epsilon_min", type=float, default=0.01, help="Minimum epsilon value")
parser.add_argument("--use_dynamic_epsilon", action="store_true", default=False,
                    help="Use dynamic epsilon-greedy strategy (default: False)")
parser.add_argument("--map_type", type=str, default="random", choices=["random", "custom"],
                    help="Map type: random (default) or custom")
parser.add_argument("--custom_holes", type=str, default="",
                    help="For custom map, provide obstacle state numbers separated by commas, e.g., '5,7,11,12'")
parser.add_argument("--use_exploration_bonus", action="store_true", default=False,
                    help="Enable exploration bonus (extra reward in non-terminal states, default: False)")
parser.add_argument("--exploration_bonus_value", type=float, default=-0.01,
                    help="Exploration bonus value in non-terminal states (e.g., -0.01)")
args = parser.parse_args()

seed = 37
random.seed(seed)
np.random.seed(seed)

# Parse custom obstacles if provided
if args.map_type == "custom" and args.custom_holes:
    try:
        custom_holes = [int(x.strip()) for x in args.custom_holes.split(',') if x.strip().isdigit()]
    except Exception as e:
        print("Error parsing custom_holes, using random map instead.")
        custom_holes = None
else:
    custom_holes = None

# Set bonus value: if exploration bonus is not enabled, set bonus to 0
bonus_value = args.exploration_bonus_value if args.use_exploration_bonus else 0.0

env = FrozenLakeEnv(grid_size=args.map_size, hole_ratio=0.25, seed=seed,
                    custom_holes=custom_holes, explore_bonus_value=bonus_value)
print("Current Environment Layout:")
env.render()

# Import and run the selected RL algorithm
if args.method == "monte_carlo":
    from monte_carlo import monte_carlo_control
    Q, rewards_list, steps_list, success_count, failure_count, time_list = monte_carlo_control(
        env, num_episodes=args.episodes, gamma=args.gamma, epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay, epsilon_min=args.epsilon_min,
        use_dynamic_epsilon=args.use_dynamic_epsilon
    )
elif args.method == "sarsa":
    from sarsa import sarsa
    Q, rewards_list, steps_list, success_count, failure_count, time_list = sarsa(
        env, num_episodes=args.episodes, alpha=0.1, gamma=args.gamma, epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay, epsilon_min=args.epsilon_min,
        use_dynamic_epsilon=args.use_dynamic_epsilon
    )
elif args.method == "q_learning":
    from q_learning import q_learning
    Q, rewards_list, steps_list, success_count, failure_count, time_list = q_learning(
        env, num_episodes=args.episodes, alpha=0.1, gamma=args.gamma, epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay, epsilon_min=args.epsilon_min,
        use_dynamic_epsilon=args.use_dynamic_epsilon
    )

policy = extract_policy(Q, env)
print("Final Policy:")
print_policy_grid(policy, env)
path = get_training_path(policy, env)

dynamic_epsilon_str = "Dynamic Epsilon: Yes" if args.use_dynamic_epsilon else "Dynamic Epsilon: No"
exploration_bonus_str = "Exploration Bonus: Yes" if args.use_exploration_bonus else "Exploration Bonus: No"
map_type_str = "Map Type: Custom" if args.map_type == "custom" else "Map Type: Random"
title = (f"Reinforcement Learning: {args.method.upper()} | Map Size: {args.map_size} | "
         f"Epsilon: {args.epsilon} | Gamma: {args.gamma} | Episodes: {args.episodes} | "
         f"{dynamic_epsilon_str} | {exploration_bonus_str} | {map_type_str}")

plt.figure(figsize=(18, 12))
plt.subplot(2, 3, 1)
plt.plot(range(1, args.episodes + 1), steps_list, color='blue', alpha=0.7)
plt.title("Number of Steps per Episode")
plt.xlabel("Episode")
plt.ylabel("Steps")

plt.subplot(2, 3, 2)
counts = [success_count, failure_count]
plt.bar(["Success", "Failure"], counts, color=['green', 'red'], alpha=0.7)
plt.title("Success vs Failure")
plt.ylabel("Count")

plt.subplot(2, 3, 3)
plot_reward_with_moving_average(rewards_list, window_size=100)
plt.title("Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")

plt.subplot(2, 3, 4)
draw_path_on_map(plt.gca(), env, path)

plt.subplot(2, 3, 5)
plot_heatmap(Q, env, plt.gca())

plt.subplot(2, 3, 6)
plt.plot(range(1, args.episodes + 1), time_list, marker='o', color='magenta')
plt.title("Cumulative Training Time")
plt.xlabel("Episode")
plt.ylabel("Time (s)")

# Wrap the title to 80 characters per line
wrapped_title = "\n".join(textwrap.wrap(title, width=80))
plt.suptitle(wrapped_title, fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])

save_dir = "./figure-result/general"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
filename = title.replace(" ", "_").replace(":", "").replace("|", "") + ".png"
filepath = os.path.join(save_dir, filename)
plt.savefig(filepath)
print(f"Figure saved to {filepath}")

plt.show()
