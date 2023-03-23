import gym
from PPO import PPO, Memory
from PIL import Image
import torch
import os

def save2Gif(max_timesteps):
    im = Image.open("0.jpg")
    images=[]
    for i in range(1, max_timesteps):
        fpath = str(i) + ".jpg"
        if os.path.exists(fpath):
            images.append(Image.open(fpath))
        else:
            break
    im.save('LunarLander.gif', save_all=True, append_images=images, loop=100, duration=15)
    for i in range(max_timesteps):
        fpath = str(i) + ".jpg"
        if os.path.exists(fpath):
            os.remove(fpath)
        else:
            break


def test():
    ############## Hyperparameters ##############
    env_name = "LunarLander-v2"
    # creating environment
    env = gym.make(env_name, render_mode='rgb_array')
    state_dim = env.observation_space.shape[0]
    action_dim = 4
    n_latent_var = 64  # number of variables in hidden layer
    lr = 0.0007
    betas = (0.9, 0.999)
    gamma = 0.99  # discount factor
    K_epochs = 4  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    #############################################

    n_episodes = 3
    max_timesteps = 300
    render = True
    save_gif = True

    filename = "PPO_{}.pth".format(env_name)

    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)

    ppo.policy_old.load_state_dict(torch.load(filename), strict=False)
    for ep in range(1, n_episodes + 1):
        ep_reward = 0
        state, _ = env.reset()
        for t in range(max_timesteps):
            action = ppo.policy_old.act(state, memory)
            state, reward, done, x, _ = env.step(action)
            ep_reward += reward

            if render:
                env.render()
            if save_gif and ep == n_episodes:
                img = env.render()
                img = Image.fromarray(img)
                img.save('{}.jpg'.format(t))
            if done:
                break

        if ep == n_episodes:
            save2Gif(max_timesteps)
        print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
        ep_reward = 0
        env.close()


if __name__ == '__main__':
    test()
