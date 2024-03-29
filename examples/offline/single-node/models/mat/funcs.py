from typing import Union
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
import torch.distributed as dist


def generate_padding_mask(max_length):
    return torch.cat(
        [
            torch.zeros(1, max_length, dtype=torch.int32),
            torch.tril(torch.ones(max_length, max_length, dtype=torch.int32)),
        ]
    )


mask = generate_padding_mask(max_length=1000).cuda()


def train_step(
    loader,
    model,
    optimizer,
    step_id: int,
    tensorboard_writer: Union[SummaryWriter, None],
):
    model.zero_grad()

    timesteps, data_batch = next(loader)
    obs = data_batch[0]
    act = data_batch[1]

    # mat inputs are expected to have an shape of [batch_size, seq_len, num_agent, dim]
    batch_size = obs.shape[0]
    obs = obs.reshape(-1, 1, obs.shape[-1])
    act = act.reshape(-1, 1, act.shape[-1])
    # print(obs.shape, act.shape)
    # print(obs.shape, data_batch)
    action_log_probs, _, _ = model(
        state=None, obs=obs, action=act, available_actions=None
    )
    loss = -(action_log_probs)
    loss = torch.sum(loss.mean(-1) * mask[timesteps].reshape(-1)) / timesteps.sum()

    # print(loss, timesteps)

    # print(loss.detach().cpu().numpy())
    model.backward(loss)
    model.step()

    if tensorboard_writer:
        tensorboard_writer.add_scalar(
            tag="training-losses", scalar_value=loss.item(), global_step=step_id
        )


def eval_step(
    model,
    eval_env,
    num_evals,
    step_id: int,
    tensorboard_writer: Union[SummaryWriter, None],
):
    model.eval()
    rsums = []
    for _ in range(num_evals):
        obs, info = eval_env.reset()
        rsum = 0
        while True:
            # print(obs)
            act, _, _ = model.get_actions(
                state=None,
                obs=torch.from_numpy(obs).cuda().reshape(-1, 1, obs.shape[-1]).float(),
                available_actions=None,
            )
            obs, reward, terminated, truncated, info = eval_env.step(
                act.detach().cpu().numpy().reshape(-1)
            )
            rsum += reward
            if terminated or truncated:
                rsums.append(rsum)
                break
    ret = np.mean(rsums)
    print(f"Evaluation step {step_id}: {ret.item()}")
    if tensorboard_writer:
        tensorboard_writer.add_scalar(
            tag="evaluation-rewards",
            scalar_value=ret.item(),
            global_step=step_id,
        )

    model.train()
    return ret
