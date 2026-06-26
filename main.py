import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim


# ==========================================
# 1. 定义环境：简单的网格世界
# ==========================================
class GridEnvironment:
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.agent_pos = [0, 0]
        self.goal_pos = [grid_size - 1, grid_size - 1]
        self.steps = 0
        self.max_steps = 50

    def reset(self):
        self.agent_pos = [0, 0]
        self.steps = 0
        return self.get_state()

    def get_state(self):
        return tuple(self.agent_pos)

    def step(self, action):
        self.steps += 1
        if action == 0:  # 上
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1:  # 下
            self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)
        elif action == 2:  # 左
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 3:  # 右
            self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)

        reward = 0
        done = False

        if self.agent_pos == self.goal_pos:
            reward = 100
            done = True
        elif self.steps >= self.max_steps:
            reward = -10
            done = True
        else:
            reward = -1

        return self.get_state(), reward, done


# ==========================================
# 2. Q 神经网络（输入维度改为 25）
# ==========================================
class QNetwork(nn.Module):
    def __init__(self, input_dim=25, output_dim=4):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),  # 输入全连接层
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)  # 输出四个动作的 Q 值
        )

    def forward(self, x):
        return self.fc(x)


# ==========================================
# 3. DQN 智能体
# ==========================================
class DQNAgent:
    def __init__(self, grid_size=5, action_size=4, lr=0.001, discount=0.99):
        self.grid_size = grid_size
        self.state_size = grid_size * grid_size  # 25 个离散状态
        self.action_size = action_size
        self.discount = discount

        # 双网络结构
        self.policy_net = QNetwork(self.state_size, action_size)
        self.target_net = QNetwork(self.state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.memory = deque(maxlen=2000)
        self.batch_size = 64

        # 探索率
        self.epsilon = 1.0
        self.epsilon_decay = 0.995  # 略微放慢衰减，让它多探索
        self.epsilon_min = 0.01

    def _state_to_one_hot(self, state):
        """核心改进：将 (row, col) 转换成 25 维的独热编码向量"""
        one_hot = np.zeros(self.state_size)
        index = state[0] * self.grid_size + state[1]
        one_hot[index] = 1.0
        return one_hot

    def remember(self, state, action, reward, next_state, done):
        # 存储时直接存转化后的独热编码
        s_oh = self._state_to_one_hot(state)
        ns_oh = self._state_to_one_hot(next_state)
        self.memory.append((s_oh, action, reward, ns_oh, done))

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        s_oh = self._state_to_one_hot(state)
        state_tensor = torch.FloatTensor(s_oh).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return torch.argmax(q_values).item()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # --- 核心修改点：在外面包裹一层 np.array() ---
        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))

        # 下面这三行保持不变（因为它们本身就是基础类型的列表，速度很快）
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        # 3. 计算当前的 Q 值
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 4. 计算目标 Q 值 (TD Target)
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.discount * max_next_q_values

        # 5. 计算损失并反向传播
        loss = self.criterion(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


# ==========================================
# 4. 训练循环（增加到 1000 回合）
# ==========================================
def train():
    env = GridEnvironment(grid_size=5)
    agent = DQNAgent(grid_size=5, action_size=4)

    episodes = 1000  # 增加回合数给深度学习模型足够的训练时间
    target_update_freq = 10

    print("开始训练 DQN 智能体...")
    print("-" * 60)
    print(f"{'回合':<8}{'总奖励':<12}{'平均损失':<12}{'当前探索率':<12}")
    print("-" * 60)

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        episode_losses = []

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)

            agent.remember(state, action, reward, next_state, done)

            loss = agent.train_step()
            if loss > 0:
                episode_losses.append(loss)

            state = next_state
            total_reward += reward

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        if (episode + 1) % target_update_freq == 0:
            agent.update_target_network()

        # 每 50 个回合打印一次明显的进度
        if (episode + 1) % 50 == 0:
            avg_loss = np.mean(episode_losses) if episode_losses else 0.0
            print(f"Ep {episode + 1:<5}{total_reward:<14.2f}{avg_loss:<14.4f}{agent.epsilon:<12.4f}")

    print("\n训练完成！")
    return agent, env


# ==========================================
# 5. 测试阶段
# ==========================================
def test(agent, env, test_episodes=5):
    print("\n测试阶段（纯利用最优策略）：")
    for episode in range(test_episodes):
        state = env.reset()
        done = False
        steps = 0

        while not done and steps < 20:
            s_oh = agent._state_to_one_hot(state)
            state_tensor = torch.FloatTensor(s_oh).unsqueeze(0)
            with torch.no_grad():
                action = torch.argmax(agent.policy_net(state_tensor)).item()

            state, _, done = env.step(action)
            steps += 1

        if state == tuple(env.goal_pos):
            print(f"测试 {episode + 1}: ✓ 成功到达目标（{steps} 步）")
        else:
            print(f"测试 {episode + 1}: ✗ 未到达目标（{steps} 步）")


if __name__ == "__main__":
    agent, env = train()
    test(agent, env)