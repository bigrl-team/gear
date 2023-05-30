## RoCE network

* Problem: torch.distributed.barriers hangs forever
    * Solution: specify NCCL_IB_GID_INDEX=<gid> in the launch script.
    * Gid can be listed with show_gids


* Problem: Cannot transition to RTR state
    * https://zhuanlan.zhihu.com/p/449803157
    * Solution: Specify 