# installation
> pip install gear

or install the rdma-version on nodes with RDMA support

> pip install gear-rdma



### Classes
* ColumnSpec

    ```python
    ColumnSpec(name: str, size: Sequence[int], dtype: torch.dtype)
    ```

    Describe the specification of a data column
    
    ``` python
    @property
    size()
    ```
        
    return the desired storage space size in bytes for the column.

* GlobalSampler series(UniformSampler, WeightedSampler, TopKSampler)
    ```python
    sync_sample(indices: torch.Tensor, weights: torch.Tensor, mpu: DeepSpeed.mpu, device: torch.Device, batch_size:int)
    ```
    make sampler aware of the global device topology and placement with DeepSpeed compatible mpu module, 


* DistributedBuffer
     ```python
    alloc(
        table_length:int, history_length: int,
        cols: Sequence[ColumnSpec]
    )
    ```
    make sampler aware of the global device topology and placement with DeepSpeed compatible mpu module, 


    ```python

    collect(indices: torch.Tensor, mpu, device: torch.Device)
    ```