from DeepRL import DQN_net,DQN
import numpy as np
import gym
from PIL import Image
import torch
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def save2Gif(max_timesteps):
    im = Image.open("0.jpg")
    images=[]
    for i in range(1, max_timesteps):
        fpath = str(i) + ".jpg"
        if os.path.exists(fpath):
            images.append(Image.open(fpath))
        else:
            break
    im.save('MountainCar-v0.gif', save_all=True, append_images=images, loop=100, duration=10)
    for i in range(max_timesteps):
        fpath = str(i) + ".jpg"
        if os.path.exists(fpath):
            os.remove(fpath)
        else:
            break

if __name__ == "__main__":
    env_name = "MountainCar-v0"
    filename_Qvalue = 'Qvalue_{}.pth'.format(env_name)
    filename_Target = 'Target_{}.pth'.format(env_name)
    env = gym.make(env_name, render_mode='rgb_array')
    hidden = 30
    lr = 0.002
    memory_capacity = 700
    gamma = 0.9
    max_episode = 20
    train_epochs = 100
    epsilon = 1.0
    max_timestep = 300
    DQN_test = DQN(env_name, max_episode, gamma, hidden, train_epochs, lr, memory_capacity, epsilon)
    DQN_test.Q_value_net.load_state_dict(torch.load(filename_Qvalue), strict = False)
    DQN_test.target_net.load_state_dict(torch.load(filename_Target), strict=False)
    for ep in range(max_episode):
        ep_reward = 0
        state, _ = env.reset()
        for t in range(max_timestep):
            action = DQN_test.choose_action(state)
            next_state, reward, done, x, _ = env.step(action)
            ep_reward += reward
            if ep == max_episode-1:
                #display_frames_as_gif(frames)
                img = env.render()
                img = Image.fromarray(img)
                img.save('{}.jpg'.format(t))
            if done:
                break
        if ep == max_episode-1:
            save2Gif(max_timestep)
        print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
        ep_reward = 0
        env.close()

