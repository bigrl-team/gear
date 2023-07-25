import torch
from .model_impl import MultiAgentTransformer
from .funcs import train_step, eval_step

default_args = {
    "n_block": 1,
    "n_embd": 64,
    "n_head": 1,
}


class Model(MultiAgentTransformer):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__(
            state_dim=0,
            obs_dim=in_dim,
            action_dim=out_dim,
            n_agent=1,
            n_block=default_args["n_block"],
            n_embd=hidden_dim,
            n_head=default_args["n_head"],
            encode_state=False,
            device=torch.device("cuda"),
            action_type="Continuous",
            dec_actor=False,
            share_actor=False,
        )


__all__ = ["train_step", "eval_step", "Model"]
