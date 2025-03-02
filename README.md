
# Frozen Lake Reinforcement Learning Solver

## II. PROBLEM STATEMENT

Consider a frozen lake with (four) holes covered by patches of very thin ice. Suppose that a robot is to glide on the frozen surface from one location (i.e., the top left corner) to another (bottom right corner) in order to pick up a frisbee there, as is illustrated in Fig. 1.

The operation of the robot has the following characteristics:
1. At a state, the robot can move in one of four directions: left, right, up, and down.
2. The robot is confined within the grid.
3. The robot receives a reward of +1 if it reaches the frisbee, -1 if it falls into a hole, and 0 for all other cases.
4. An episode ends when the robot reaches the frisbee or falls into a hole.

## III. REQUIREMENT

### A. What to be done

Three tasks, with the following mark weightage, are to be completed for this project:

**Task 1: Basic Implementation (25%)**

Write a Python program to compute an optimal policy for the Frozen Lake problem described above using the following three tabular (i.e., not involving any use of a neural network) reinforcement learning techniques:
1. First-visit Monte Carlo control without exploring starts.
2. SARSA with an epsilon-greedy behavior policy.
3. Q-learning with an epsilon-greedy behavior policy.

You may choose appropriate values for parameters such as the discount factor, learning rate, etc.

**Task 2: Extended Implementation (25%)**

Increase the grid size to at least 10×10 while maintaining the same proportion between the number of holes and the number of states (i.e., 4/16 = 25%). Distribute the holes randomly without completely blocking access to the frisbee, and then repeat Task 1.

---

## Project Introduction

This project is designed to solve the Frozen Lake problem using three classical reinforcement learning algorithms. The environment is modeled as a grid where the robot (agent) must navigate from the top-left corner to the bottom-right corner, avoiding holes and reaching the goal to pick up a frisbee. The reward structure is as follows:
- **+1** for reaching the frisbee (goal).
- **-1** for falling into a hole.
- **0** for all other transitions (unless an exploration bonus is used).

The project is modularized into multiple Python files:
- **env.py**: Defines the Frozen Lake environment.
- **utils.py**: Provides helper functions (epsilon-greedy action selection, Q-table initialization, plotting helpers, policy extraction, etc.).
- **monte_carlo.py**: Implements the First-Visit Monte Carlo Control algorithm.
- **sarsa.py**: Implements the SARSA algorithm.
- **q_learning.py**: Implements the Q-learning algorithm.
- **main.py**: The main script for parsing command-line arguments, running training, plotting, and saving figures.

---

## 1. Algorithms: Principles and Pseudocode Diagrams

### 1.1. First-Visit Monte Carlo Control
**Principle:**  
This method collects complete episodes using an epsilon-greedy policy, then updates the Q-table by averaging returns for first visits of state-action pairs.

**Pseudocode:**
```
Initialize Q(s, a) arbitrarily for all states s and actions a.
For each episode:
    Generate an episode: s0, a0, r1, s1, a1, r2, ..., s_T.
    For each (s, a) in the episode (first visit only):
         G = sum_{t=k}^{T} (gamma^(t-k)) * r_{t+1}
         Update Q(s, a) as average of returns.
```

### 1.2. SARSA (On-Policy)
**Principle:**  
SARSA updates the Q-value based on the current policy’s action selection. The update uses the immediate reward plus the discounted Q-value of the next state-action pair.

**Pseudocode:**
```
Initialize Q(s, a) arbitrarily for all s and a.
For each episode:
    Initialize s, choose a using epsilon-greedy policy.
    For each step in the episode:
         Take action a, observe r and s'.
         Choose a' using epsilon-greedy from Q(s', ·).
         Update Q(s, a) = Q(s, a) + alpha * [r + gamma * Q(s', a') - Q(s, a)]
         s <- s', a <- a'
         If s' is terminal, break.
```

### 1.3. Q-Learning (Off-Policy)
**Principle:**  
Q-Learning uses the maximum Q-value for the next state (irrespective of the current policy) to update the current state-action value.

**Pseudocode:**
```
Initialize Q(s, a) arbitrarily for all s and a.
For each episode:
    Initialize s.
    For each step:
         Choose action a using epsilon-greedy policy.
         Take action a, observe r and s'.
         Update Q(s, a) = Q(s, a) + alpha * [r + gamma * max_{a'} Q(s', a') - Q(s, a)]
         s <- s'
         If s' is terminal, break.
```

---

## 2. Map Generation Principles

The environment supports two map generation modes:
- **Random Map Mode:**  
  Obstacles (holes) are randomly generated based on a fixed ratio (e.g., 25% of the states are holes). The algorithm ensures that there is at least one valid path from the start to the goal.

- **Custom Map Mode:**  
  The user can provide specific state numbers for obstacles using the command-line parameter `--custom_holes`. In this mode, the given holes are used directly (with a check to ensure the start and goal are not included).

---

## 3. Command-Line Examples for Basic Implementation

For the basic implementation, dynamic epsilon and exploration bonus are NOT used. The map is 4×4, and the parameters include a custom set of obstacles.

Example (using custom holes "5,10,15,20", epsilon 0.1, gamma 0.9):
```bash
python main.py --method sarsa --map_size 4 --epsilon 0.1 --gamma 0.9 --episodes 5000 --map_type custom --custom_holes "5,10,15,20"
```
*Explanation:*  
- **--method sarsa**: Uses the SARSA algorithm.
- **--map_size 4**: Uses a 4×4 grid.
- **--epsilon 0.1**, **--gamma 0.9**: Sets the initial epsilon and discount factor.
- **--episodes 5000**: Runs 5000 episodes.
- **--map_type custom** and **--custom_holes "5,10,15,20"**: Specifies a custom map with obstacles at states 5, 10, 15, and 20.
  
(Note: In this basic implementation, dynamic epsilon and exploration bonus options are off by default.)

---

## 4. Improved Algorithm (Advanced Features)

In the improved version, the following features are added:
- **Dynamic Epsilon-Greedy Strategy:**  
  Epsilon decays over episodes based on parameters `--epsilon_decay` and `--epsilon_min`. For example, the current epsilon is computed as:
  ```
  current_epsilon = max(epsilon_min, epsilon * (epsilon_decay^episode))
  ```
- **Exploration Bonus:**  
  In non-terminal states, a small bonus (e.g., -0.01) is applied to encourage the agent to finish the episode faster. This is set using the `--use_exploration_bonus` flag and controlled via `--exploration_bonus_value`.

### Detailed Command-Line Parameter Explanation (Advanced Version)
- `--method`: Algorithm to use (`sarsa`, `monte_carlo`, or `q_learning`).
- `--map_size`: Size of the grid (e.g., 4, 6, 10).
- `--epsilon`: Initial epsilon value.
- `--gamma`: Discount factor.
- `--episodes`: Number of training episodes.
- `--epsilon_decay`: The decay rate of epsilon per episode.
- `--epsilon_min`: The minimum epsilon value.
- `--use_dynamic_epsilon`: If provided, epsilon decays over episodes.
- `--map_type`: Map type; "random" for a random map or "custom" for a user-defined map.
- `--custom_holes`: A comma-separated list of state numbers to use as obstacles when using a custom map.
- `--use_exploration_bonus`: If provided, enables an exploration bonus in non-terminal states.
- `--exploration_bonus_value`: The bonus (typically a small negative value) applied in non-terminal states.

### Advanced Command-Line Example:
```bash
python main.py --method q_learning --map_size 10 --epsilon 0.2 --gamma 0.95 --episodes 2000 --epsilon_decay 0.99 --epsilon_min 0.01 --use_dynamic_epsilon --map_type custom --custom_holes "15,22,35,48" --use_exploration_bonus --exploration_bonus_value -0.01
```
*Explanation:*  
Runs Q-Learning on a 10×10 custom grid with obstacles (states 15, 22, 35, 48), using a dynamic epsilon starting at 0.2 (decaying with rate 0.99 to a minimum of 0.01), and applies an exploration bonus of -0.01 in non-terminal states.

---

## 5. Detailed Description of the Figures

The program generates a figure composed of six subplots:

1. **Number of Steps per Episode:**  
   Plots the number of steps taken in each episode. This graph helps visualize the learning progress in terms of how quickly the agent finishes episodes.

2. **Success vs. Failure Counts:**  
   A bar chart that displays the total number of successful episodes (reaching the goal) versus failed episodes (falling into a hole).

3. **Reward per Episode (with Moving Average):**  
   A scatter plot of the raw reward for each episode, overlaid with a moving average (e.g., MA(100)) to show the trend of rewards over time.

4. **Training Path:**  
   Displays the grid with obstacles, marking the start and goal with the words "Start" and "Goal" (in distinct colors) and highlighting the path taken by the agent using directional arrows.

5. **Action Value Heatmap:**  
   For each grid cell, the Q-values for the four possible actions are displayed in a square divided into four triangles (diagonally). Each triangle is color-coded from blue (low) to red (high) and shows the corresponding Q-value.

6. **Cumulative Training Time:**  
   A plot showing the cumulative training time (in seconds) versus episodes, which helps gauge the efficiency of the training process.

---

## Additional Details

- **Modularity:**  
  The project is organized into several Python files for better maintainability and modularity.
  
- **Output:**  
  The program prints the environment layout, episode statistics, and saves a figure (with a wrapped title) to the `./figure-result` directory. The filename is automatically generated from the experiment title (illegal filename characters are removed).

- **Assumptions:**  
  All algorithms use tabular methods (no neural networks). The parameters such as learning rate (alpha), discount factor (gamma), epsilon settings, etc., are set via command-line arguments.

---

## Command-Line Examples Summary (Without Advanced Features)

For a basic run (without dynamic epsilon and exploration bonus) on a 4×4 grid using custom obstacles:
```bash
python main.py --method sarsa --map_size 4 --epsilon 0.1 --gamma 0.9 --episodes 5000 --map_type custom --custom_holes "5,10,15,20"
```

For an advanced run (with dynamic epsilon and exploration bonus) on a 10×10 grid:
```bash
python main.py --method q_learning --map_size 10 --epsilon 0.2 --gamma 0.95 --episodes 2000 --epsilon_decay 0.99 --epsilon_min 0.01 --use_dynamic_epsilon --map_type custom --custom_holes "15,22,35,48" --use_exploration_bonus --exploration_bonus_value -0.01
```

---

## License

This project is provided for educational purposes only.

---

# 中文翻译

## II. 问题陈述

考虑一个冰冻湖，其上覆盖着（四个）由非常薄冰面覆盖的冰洞。假设一个机器人需要在冰冻表面上滑行，从一个位置（即左上角）到另一个位置（右下角），以便在那里拾取一个飞盘，如图 1 所示。

机器人的操作具有以下特点：
1. 在某一状态下，机器人可以向左、向右、向上或向下移动。
2. 机器人被限制在网格内移动。
3. 如果机器人到达飞盘位置，获得 +1 奖励；如果掉进冰洞，则获得 -1 奖励；其他情况下奖励为 0。
4. 当机器人到达飞盘或掉进冰洞时，一个回合结束。

## III. 要求

### A. 任务说明

本项目共需完成以下三个任务，各任务的分值比例如下：

**任务 1：基础实现 (25%)**

编写一个 Python 程序，使用以下三种表格（即不涉及神经网络）强化学习技术，为上述冰冻湖问题（参见问题陈述）计算出最优策略：
1. 无探索起始的首访蒙特卡罗控制。
2. 使用 epsilon-greedy 行为策略的 SARSA 算法。
3. 使用 epsilon-greedy 行为策略的 Q-learning 算法。你可以选择所有必要参数（例如折扣率、学习率等）的值。

**任务 2：扩展实现 (25%)**

将网格尺寸增大到至少 10×10，同时保持冰洞数目与状态数之间的比例（例如 4/16 = 25%）。随机分布冰洞，但不要完全阻挡通往飞盘的通道。然后重复任务 1。

---

## 项目介绍

本项目旨在使用三种经典的强化学习算法解决冰冻湖问题。环境被建模为一个网格，其中机器人（智能体）需要从左上角滑行到右下角，避免冰洞并最终达到目标以拾取飞盘。奖励结构如下：
- 到达目标（飞盘）：+1
- 掉进冰洞：-1
- 其他状态：0（若启用了探索奖励，则会有一个较小的负奖励）

项目采用模块化设计，共包含多个 Python 文件：
- **env.py**：定义冰冻湖环境类。
- **utils.py**：提供辅助函数（如 epsilon-greedy 策略、Q 表初始化、策略打印、绘图工具等）。
- **monte_carlo.py**：实现首访蒙特卡罗控制算法。
- **sarsa.py**：实现 SARSA 算法。
- **q_learning.py**：实现 Q-learning 算法。
- **main.py**：主程序，用于解析命令行参数、设置环境和算法、运行训练、绘图并保存图像。

---

## 1. 算法原理及伪代码图

### 1.1. 首访蒙特卡罗控制
**原理：**  
该方法通过 epsilon-greedy 策略生成完整回合，并对回合中第一次访问的状态-动作对计算累计回报，再更新 Q 表。

**伪代码：**
```
Initialize Q(s, a) arbitrarily.
For each episode:
    Generate an episode: s0, a0, r1, s1, a1, r2, ..., s_T.
    For each (s, a) in the episode (first visit only):
         G = sum_{t=k}^{T} (gamma^(t-k)) * r_{t+1}
         Update Q(s, a) as the average of returns.
```

### 1.2. SARSA (在策略)
**原理：**  
SARSA 使用当前策略选取动作，并依据当前状态、动作及下一个状态-动作对来更新 Q 值。

**伪代码：**
```
Initialize Q(s, a) arbitrarily.
For each episode:
    Initialize s, choose a using epsilon-greedy.
    For each step:
         Take action a, observe r and s'.
         Choose a' using epsilon-greedy from Q(s', ·).
         Update Q(s, a) = Q(s, a) + alpha * [r + gamma * Q(s', a') - Q(s, a)]
         s <- s', a <- a'
         If s' is terminal, break.
```

### 1.3. Q-learning (离策略)
**原理：**  
Q-learning 采用下一状态的最大 Q 值（与当前策略无关）来更新当前状态-动作的 Q 值。

**伪代码：**
```
Initialize Q(s, a) arbitrarily.
For each episode:
    Initialize s.
    For each step:
         Choose action a using epsilon-greedy.
         Take action a, observe r and s'.
         Update Q(s, a) = Q(s, a) + alpha * [r + gamma * max_{a'} Q(s', a') - Q(s, a)]
         s <- s'
         If s' is terminal, break.
```

---

## 2. 地图生成原理

本环境支持两种地图生成模式：
- **随机地图模式：**  
  根据固定比例（例如 25%）随机生成冰洞，并确保至少存在一条从起点到目标的有效路径。
- **自设地图模式：**  
  用户可以通过命令行参数 `--custom_holes` 指定障碍状态编号（例如 "5,7,11,12"），此时使用自定义障碍。

---

## 3. 命令行示例（基础实现）

在基础实现中，不启用动态 epsilon 以及探索奖励。地图尺寸为 4×4，参数示例：
```bash
python main.py --method sarsa --map_size 4 --epsilon 0.1 --gamma 0.9 --episodes 5000 --map_type custom --custom_holes "5,10,15,20"
```
*说明：*  
- 使用 SARSA 算法；  
- 地图为 4×4 自设地图，障碍位于状态 5, 10, 15, 20；  
- 初始 epsilon 为 0.1，gamma 为 0.9，训练 5000 个回合。

---

## 4. 改进算法说明（高级实现）

在改进算法中，我们引入以下新功能：
- **动态 Epsilon-greedy 策略：**  
  Epsilon 随着训练回合数逐渐衰减，其计算公式为：  
  ```
  current_epsilon = max(epsilon_min, epsilon * (epsilon_decay^episode))
  ```
- **探索奖励：**  
  在非终点状态下，为鼓励智能体尽快结束回合，给予一个小的负奖励（例如 -0.01）。此功能由命令行参数 `--use_exploration_bonus` 和 `--exploration_bonus_value` 控制。

### 高级命令行参数说明
- `--method`: RL algorithm to use (`sarsa`, `monte_carlo`, `q_learning`).
- `--map_size`: Grid size (e.g., 4, 6, 10).
- `--epsilon`: Initial epsilon value.
- `--gamma`: Discount factor.
- `--episodes`: Number of training episodes.
- `--epsilon_decay`: Epsilon decay rate per episode.
- `--epsilon_min`: Minimum epsilon value.
- `--use_dynamic_epsilon`: If provided, epsilon decays over episodes.
- `--map_type`: Map type; "random" for random map or "custom" for a user-defined map.
- `--custom_holes`: For a custom map, specify obstacle state numbers separated by commas (e.g., "5,7,11,12").
- `--use_exploration_bonus`: If provided, enables an exploration bonus in non-terminal states.
- `--exploration_bonus_value`: The bonus value in non-terminal states (e.g., -0.01).

### Advanced Command-Line Example
```bash
python main.py --method q_learning --map_size 10 --epsilon 0.2 --gamma 0.95 --episodes 2000 --epsilon_decay 0.99 --epsilon_min 0.01 --use_dynamic_epsilon --map_type custom --custom_holes "15,22,35,48" --use_exploration_bonus --exploration_bonus_value -0.01
```
*说明：*  
在 10×10 自设地图上运行 Q-learning，其中障碍为状态 15, 22, 35, 48。使用动态 epsilon（初始 0.2，衰减至最小值 0.01）和探索奖励（非终点状态奖励 -0.01）。

---

## 5. Detailed Description of the Figures

The program generates a figure containing six subplots:

1. **Number of Steps per Episode:**  
   Plots the number of steps taken in each episode. It shows the learning progress in terms of episode length.

2. **Success vs. Failure Counts:**  
   A bar chart that shows the number of successful episodes (reaching the frisbee) versus failed episodes (falling into a hole).

3. **Reward per Episode (with Moving Average):**  
   A scatter plot of raw rewards per episode overlaid with a moving average (e.g., MA(100)). This indicates the trend of rewards as training progresses.

4. **Training Path:**  
   Displays the grid environment with obstacles, marking the start with "Start" (orange) and the goal with "Goal" (purple). The path taken by the agent is indicated with directional arrows.

5. **Action Value Heatmap:**  
   For each grid cell, the Q-values for the four actions are visualized. Each cell is divided into four triangles (each corresponding to an action), with colors ranging from blue (low) to red (high). The Q-value is shown in each triangle.

6. **Cumulative Training Time:**  
   A line plot showing the cumulative training time (in seconds) across episodes, which provides insight into the computational efficiency.

---

## Additional Project Details

- **Modular Design:**  
  The project is split into multiple Python files (env.py, utils.py, monte_carlo.py, sarsa.py, q_learning.py, main.py) for clarity and maintainability.

- **Output:**  
  The program prints the environment layout and per-episode training statistics to the console. It then displays a figure containing the six subplots described above. The figure is also saved in the `./figure-result` directory, with the filename based on the experiment title (wrapped to 80 characters per line).

---

## Command-Line Examples Summary

### Basic Implementation (No dynamic epsilon, no exploration bonus; 4×4 custom map)
```bash
python main.py --method sarsa --map_size 4 --epsilon 0.1 --gamma 0.9 --episodes 5000 --map_type custom --custom_holes "5,10,15,20"
```

### Advanced Implementation (Dynamic epsilon and exploration bonus enabled)
```bash
python main.py --method q_learning --map_size 10 --epsilon 0.2 --gamma 0.95 --episodes 2000 --epsilon_decay 0.99 --epsilon_min 0.01 --use_dynamic_epsilon --map_type custom --custom_holes "15,22,35,48" --use_exploration_bonus --exploration_bonus_value -0.01
```

---

## License

This project is provided for educational purposes only.
```

---

This **README.md** file is as detailed as possible, covering the assignment problem statement, requirements, project introduction, algorithm principles with pseudocode, map generation principles, command-line examples for both basic and advanced implementations, and a detailed description of the six figures. The latter half of the document is provided in Chinese as a translation.