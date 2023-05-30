from dataclasses import dataclass


@dataclass
class TrajectoryInterfaceParams:
    """
    This class is the universal param configuration class for TrajectoryInterface instances(including both writer and reader). To fully express the multi-level parallelism, the param class contains multiple world/rank attributes pair, which might be confusing for users. Their roles are explained as follows:

    TrajectoryInterfaceParams:
        * node_world: Critical in ***cross-node*** cases, denotes the total number of compute nodes involved.
        * node_rank: Critical in ***cross-node*** cases, denotes the rank of the node in the node group(e.g. 0 for node 0, 1 for node 1 and etc.)
        * shard_world: Multiple shards may exist within a single compute node, and hence shard_world denotes the number of shards on the local node.
        * shard_rank: rank of shard within local shard group.
    """

    node_world: int
    node_rank: int
    node_key: int
    shard_world: int
    shard_rank: int
    shard_key: int
    shard_capacity: int
    shard_group_world: int
    shard_group_rank: int
