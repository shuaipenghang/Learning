import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt
import gym
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#存储训练数据
class Memory:
    def __init__(self):
        self.action = []
        self.states = []
        self.next_states = []
        self.rewards = []

    def clear_memory(self):
        del self.action[:]
        del self.states[:]
        del self.next_states[:]
        del self.rewards[:]

class DQN_net(nn.Module):
    def __init__(self, state_dim, action_dim, hidden):
        super(DQN_net, self).__init__()
        self.h1 = nn.Linear(state_dim, hidden)
        self.h1.weight.data.normal_(0, 0.1)
        self.h2 = nn.Linear(hidden, action_dim)
        self.h2.weight.data.normal_(0, 0.1)

    #前向传播
    def forward(self, x):
        x = self.h1(x)
        x = F.relu(x)
        x = self.h2(x)

        return x

class DQN:
    def __init__(self, env_name, max_episode, gamma, hidden, train_epochs, lr, memory_capacity, epsilon):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.max_episode = max_episode
        self.lr = lr #网络的学习率
        self.gamma = gamma
        self.memory_capacity = memory_capacity #记忆容量
        self.epsilon = epsilon
        self.train_epochs = train_epochs #每隔多少次训练一次
        self.memory_counter = 0
        self.learn_counter = 0
        self.batch = 32

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.Memory = Memory() #存储训练数据
        self.Q_value_net = DQN_net(self.state_dim, self.action_dim, hidden).to(device) #Q表训练网络
        self.target_net = DQN_net(self.state_dim, self.action_dim, hidden).to(device)
        self.optimizer = torch.optim.Adam(self.Q_value_net.parameters(), lr=lr)
        self.loss = nn.MSELoss() #损失函数


    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0).to(device)
        if np.random.randn() <= self.epsilon:
            action_value = self.Q_value_net.forward(state)
            action = torch.max(action_value, 1)[1].data.cpu().numpy()
            action = action[0]
        else:
            action = np.random.randint(0, self.action_dim)

        return action

    def memory_store(self, state, next_state, action, reward):
        if self.memory_counter % 500 == 0: #每500次清洗一次数据，最终一直停留在这
            print("The experience pool collects {} time experience".format(self.memory_counter))

        self.Memory.next_states.append(next_state)
        self.Memory.action.append(action)
        self.Memory.states.append(state)
        self.Memory.rewards.append(reward)
        self.memory_counter += 1

    def learn(self):
        if self.learn_counter % self.train_epochs == 0:
            self.target_net.load_state_dict(self.Q_value_net.state_dict()) #每train_epochs次更新target_net
            self.learn_counter = 0
        self.learn_counter += 1
        sample_index = np.random.choice(len(self.Memory.states), self.batch)
        batch_states = torch.FloatTensor(np.array(self.Memory.states)[sample_index]).to(device)
        batch_action = torch.LongTensor(np.array(self.Memory.action)[sample_index].astype(int)).to(device)
        batch_next_states = torch.FloatTensor(np.array(self.Memory.next_states)[sample_index]).to(device)
        batch_rewards = torch.FloatTensor(np.array(self.Memory.rewards)[sample_index]).to(device)
        #print('batch_states\n', batch_states.size())
        #print('batch_action\n', batch_action.size())
        q_eval = self.Q_value_net(batch_states).gather(1, batch_action.view(self.batch, 1))
        q_next = self.target_net(batch_next_states).detach()
        #print('batch_rewards\n', batch_rewards.size())
        #print('q_next\n', q_next.max(1)[0].size())
        q_target = batch_rewards + self.gamma*q_next.max(1)[0]#.view(self.batch, 2)

        loss = self.loss(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def train(self):
        step_counter_list = []
        for episode in range(self.max_episode):
            state, _ = self.env.reset()
            step_counter = 0
            while True:
                step_counter += 1
                self.env.render()
                action = self.choose_action(state)
                next_state, reward, done, info, _ = self.env.step(action)
                reward = reward * 100 if reward > 0 else reward * 5
                #print('state:',state,'action:',action,'next_state:',next_state,'reward:',reward)
                self.memory_store(state, next_state, action, reward)
                if self.memory_counter >= self.memory_capacity: #此处永远无法到达
                    self.learn()
                    if done:
                        print("episode {}, the reward is {}".format(episode, round(reward, 3)))
                if done:
                    print("This is {} episode".format(episode))
                    break

                state = next_state
            if episode == self.max_episode-1: #最后一轮
                print("This experiment is done!")
                torch.save(self.Q_value_net.state_dict(), './Qvalue_{}.pth'.format(self.env_name))
                torch.save(self.target_net.state_dict(), './Target_{}.pth'.format(self.env_name))

if __name__ == "__main__":
    env_name = "MountainCar-v0"
    hidden = 30
    lr = 0.002
    memory_capacity = 700
    gamma = 0.9
    max_episode = 100
    train_epochs = 100
    epsilon = 0.9
    DQN_test = DQN(env_name, max_episode, gamma, hidden, train_epochs, lr, memory_capacity, epsilon)
    DQN_test.train()
