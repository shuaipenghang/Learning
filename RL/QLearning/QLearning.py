import gym
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from matplotlib import animation

def Q_table_init(state_dim, action_dim):
    #初始化Q表
    Q_table = np.random.uniform(low = -1, high = 1, size = (state_dim, action_dim))

    return Q_table
def traverse_imgs(writer, images):
    # 遍历所有图片，并且让writer抓取视频帧
    for img in images:
        plt.imshow(img)
        writer.grab_frame()
        plt.pause(0.01)
        plt.clf()

def save2Gif(frames, game_name):
    plt.figure(figsize=(frames[0].shape[1]/72.0, frames[0].shape[0]/72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    plt.ion()  # 为了可以动态显示
    plt.tight_layout()
    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    
    anim.save(game_name+'.gif')

class QLearning:
    def __init__(self, env_name, Q_table, alpha, dis_factor, max_episode, max_timestep, save_gif):
        self.env_name = env_name #构建环境
        self.env = gym.make(self.env_name, render_mode='rgb_array')
        self.Q_table = Q_table #Q表
        self.alpha = alpha #学习率
        self.dis_factor = dis_factor #学习折扣
        self.max_episode = max_episode #最大迭代次数
        self.max_timestep = max_timestep
        self.save_gif = save_gif

    def get_action(self, state, episode):
        prob = np.random.rand()
        epsilon = 0.5 * (1 / (episode + 1))
        if prob < epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q_table[state])

    def update(self): #更新一次迭代的Q表
        frames = []
        for episode in range(self.max_episode):
            done = False
            state = self.env.reset()[0]
            reward = 0
            Reward_cum = 0 #累计奖励
            action = self.get_action(state, episode)
            for t in range(self.max_timestep):
                #选择动作
                state_next, reward, done, info, _ = self.env.step(action) #更新动作
                #更新Q表
                self.Q_table[state, action] += \
                    self.alpha*(reward + self.dis_factor*np.max(self.Q_table[state_next]) - self.Q_table[state, action])
                Reward_cum += reward #更新奖励
                state = state_next
                action = self.get_action(state, episode)

                if save_gif and episode == max_episode - 1:
                    frames.append(self.env.render())
                if done:
                    break

            if episode == max_episode - 1:
                print('图片已保存')
                save2Gif(frames, self.env_name)
            if episode % 50 == 0:
                print('episode:{}:total reward:{}'.format(episode, Reward_cum))

        return self.Q_table


if __name__ == '__main__':
    # 首先需要构建Q表，Q表是状态与动作的对应
    env_name = "Taxi-v3"
    env = gym.make(env_name)
    state_dim = env.observation_space.n
    action_dim = env.action_space.n
    print('状态空间大小为', state_dim)
    print('动作空间大小为', action_dim)
    Q_table = Q_table_init(state_dim, action_dim)
    print("初始化Q表为\n", Q_table)
    alpha = 0.5
    dis_factor = 0.99
    max_episode = 3000
    max_timestep = 300
    save_gif = True
    QL = QLearning(env_name, Q_table, alpha, dis_factor, max_episode, max_timestep, save_gif)
    next_Q_table = QL.update()
    np.savetxt('Qvalue.txt', next_Q_table)