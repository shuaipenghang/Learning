import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.action = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = [] #判断结束

    def clear_memory(self):
        del self.action[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        #actor，输出动作的概率
        self.action_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, action_dim),
            nn.Softmax(dim = -1)
        )
        #critic
        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, 1) #输出一个value值, R-value
        )

    def forward(self):
        raise NotImplementedError #抛出异常

    #获得动作
    def act(self, state, memory):
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)  #输出概率值
        action = dist.sample()

        memory.states.append(state)
        memory.action.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()

    #
    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs) #换成概率

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy() #计算熵

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value) , dist_entropy #squeeze为维度压缩

class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        try:
            rewards = (rewards - rewards.means()) / (rewards.std() + 1e-5)
        except AttributeError:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            print('rewards的类型为\n', type(rewards))

        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.action).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        for _ in range(self.K_epochs):
            logprobs, state_value, dist_entropy = self.policy.evaluate(old_states, old_actions)
            ratios = torch.exp(logprobs - old_logprobs.detach()) #detach()切断反向传播

            advantages = rewards - state_value.detach() #r-b
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr2, surr1) + 0.5*self.MseLoss(state_value, rewards) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward() #反向传播
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())



def main():
    env_name = "LunarLander-v2"
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = 4
    render = True
    solved_reward = 230
    log_interval = 20
    max_episodes = 50000
    max_timestep = 300
    n_latent_var = 64
    update_timestep = 2000
    lr = 0.002
    betas = (0.9, 0.999)
    gamma = 0.99
    K_epochs = 4
    eps_clip = 0.2 #clip PPO
    random_seed = None

    n_episodes = 3
    save_gif = False
    filename = "PPO_{}.path".format(env_name)
    directory = "./preTrained/"
    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)

    memory = Memory() #存放数据
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)

    running_reward = 0
    avg_length = 0
    timestep = 0

    for i_episode in range(1, max_episodes+1):
        state, _ = env.reset() #初始化
        for t in range(max_timestep):
            timestep += 1
            action = ppo.policy_old.act(state, memory)
            state, reward, done, info, _ = env.step(action)

            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            #
            if timestep % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                timestep = 0

            running_reward += reward
            if render:
                env.render()
            if done:
                break

        avg_length += t

        if running_reward > (log_interval * solved_reward):
            print('########## Solved! ##########')
            torch.save(ppo.policy.state_dict(), './PPO_{}.pth'.format(env_name))
            break

        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))

            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0

if __name__ == '__main__':
    main()

