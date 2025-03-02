from collections import defaultdict
import time
import random
import numpy as np
from utils import epsilon_greedy_action, initialize_Q


def sarsa(env, num_episodes=5000, alpha=0.1, gamma=0.9, epsilon=0.1,
          epsilon_decay=0.99, epsilon_min=0.01, use_dynamic_epsilon=False,
          invalid_penalty=-1):
    """
    SARSA (on-policy) algorithm:
    - Dynamically determine the current epsilon value if enabled.
    - Record the cumulative reward, steps, and result (Success/Failure/Unknown based on the last reward) per episode.
    - Also record the cumulative training time (in seconds).
    Note: For invalid moves (i.e., out-of-bound actions), a penalty of -1 is applied to update the Q value.
    """
    Q = initialize_Q(env)
    rewards_list = []
    steps_list = []
    success_count = 0
    failure_count = 0
    cumulative_time = 0.0
    time_list = []

    for episode in range(num_episodes):
        current_epsilon = max(epsilon_min, epsilon * (epsilon_decay ** episode)) if use_dynamic_epsilon else epsilon
        start_time_episode = time.time()
        state = env.reset()
        action = epsilon_greedy_action(Q, state, current_epsilon)
        done = False
        episode_reward = 0
        step_count = 0
        episode_list = []

        while not done:
            next_state, reward, done, valid_move = env.step(action)
            if valid_move:
                episode_list.append((state, action, reward))
                episode_reward += reward
                step_count += 1
                if done:
                    Q[state][action] += alpha * (reward - Q[state][action])
                    break
                next_action = epsilon_greedy_action(Q, next_state, current_epsilon)
                Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
                state = next_state
                action = next_action
            else:
                Q[state][action] += alpha * (invalid_penalty - Q[state][action])
                action = epsilon_greedy_action(Q, state, current_epsilon)
                continue
            if done:
                break
        if use_dynamic_epsilon == False:
            episode_reward = max(-1, min(1, episode_reward))
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
        print(
            f"Episode {episode + 1}: steps = {step_count}, Reward = {episode_reward}, current_epsilon = {current_epsilon:.4f}, {result_str}")
        episode_time = time.time() - start_time_episode
        cumulative_time += episode_time
        time_list.append(cumulative_time)
    return Q, rewards_list, steps_list, success_count, failure_count, time_list


if __name__ == '__main__':
    pass
