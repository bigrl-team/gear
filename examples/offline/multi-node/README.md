# Multi-Node Offline Training Example
This folder includes a minimal example for multi-node offline training.





## Dependencies
Before getting your hands dirty, please double check the following dependency requirements are met on your target systems.

### `PDSH` as Multi-node Launchers
In the current state, GEAR provide distributed training interface with DeepSpeed, which relies on ``pdsh`` tool for distributed task launching. To install ``pdsh`` with *APT*:
```shell
$ apt install pdsh -y
```