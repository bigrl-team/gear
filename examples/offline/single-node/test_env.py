import gym
import torch
import torch.nn as nn
from torch.distributions import Normal


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim) -> None:
        super().__init__()
        self.mean_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 3),
            nn.Tanh(),
        )
        

        self.sigma_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 3),
        )

    def forward(self, x):
        mean, sigma = self.mean_net(x), torch.exp(self.sigma_net(x))
        return mean, sigma


env = gym.make("Hopper-v4")


model = MLP(in_dim=11, hidden_dim=64, out_dim=8)


def eval(num_eval_trajectories: int):
    model.eval()
    rsums = []
    for i in range(num_eval_trajectories):
        obs = env.reset()
        rsum = 0
        for i in range(10000):
            obs = obs.reshape(1, *obs.shape)
            mean, sigma = model(torch.from_numpy(obs).float())
            print(mean.shape, sigma.shape)
            distri = Normal(mean, sigma)
            act = distri.rsample()
            print(act.shape)
            obs, reward, done, info = env.step(act.detach().numpy().squeeze(0))
            print(reward)
            rsum += reward
            if done:
                print(f"total steps {i}, reward {rsum}")
                rsums.append(rsum)
                break


if __name__ == "__main__":
    eval(10)
