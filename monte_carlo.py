from collections import defaultdict
import time
import random
import numpy as np
from utils import epsilon_greedy_action, initialize_Q

def monte_carlo_control(env, num_episodes=5000, gamma=0.9, epsilon=0.1,
                        epsilon_decay=0.99, epsilon_min=0.01, use_dynamic_epsilon=False):
    """
    First-visit Monte Carlo Control:
    - Determines the current epsilon based on whether dynamic epsilon is enabled.
    - Records the cumulative reward, steps, and success/failure per episode (judged by the last reward).
    - Also records cumulative training time (in seconds).
    """
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    Q = initialize_Q(env)
    rewards_list = []
    steps_list = []
    success_count = 0
    failure_count = 0
    cumulative_time = 0.0
    time_list = []

    for episode in range(num_episodes):
        start_time_episode = time.time()
        current_epsilon = max(epsilon_min, epsilon * (epsilon_decay ** episode)) if use_dynamic_epsilon else epsilon

        episode_list = []
        state = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        while not done:
            action = epsilon_greedy_action(Q, state, current_epsilon)
            next_state, reward, done = env.step(action)
            episode_list.append((state, action, reward))
            episode_reward += reward
            step_count += 1
            state = next_state

        rewards_list.append(episode_reward)
        steps_list.append(step_count)
        if episode_list:
            last_reward = episode_list[-1][2]
        else:
            last_reward = 0
        if last_reward == 1:
            success_count += 1
            result_str = "Success"
        elif last_reward == -1:
            failure_count += 1
            result_str = "Failure"
        else:
            result_str = "Unknown"
        print(f"Episode {episode+1}: steps = {step_count}, {result_str}, current_epsilon = {current_epsilon:.4f}")

        # Update Q values
        sa_in_episode = set()
        for i, (s, a, _) in enumerate(episode_list):
            if (s, a) not in sa_in_episode:
                sa_in_episode.add((s, a))
                G = 0.0
                power = 0
                for (_, _, r) in episode_list[i:]:
                    G += (gamma ** power) * r
                    power += 1
                returns_sum[(s, a)] += G
                returns_count[(s, a)] += 1.0
                Q[s][a] = returns_sum[(s, a)] / returns_count[(s, a)]
        episode_time = time.time() - start_time_episode
        cumulative_time += episode_time
        time_list.append(cumulative_time)
    return Q, rewards_list, steps_list, success_count, failure_count, time_list

if __name__ == '__main__':
    pass
