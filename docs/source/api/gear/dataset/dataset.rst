``gear.dataset``
===============================


.. py:currentmodule:: gear.dataset


.. autoclass:: SharedDataset

    .. note:: ``gear.dataset.SharedDataset`` serves as an intermediary data structure when building training dataset within a single process or an interface for sharing offline dataset. While its instances allow runtime updates on the trajectory data, concurrent read/write operations on the same trajectory data block could lead to potential data corruption. If you are interested in its variants that can safely share/update trajectory data, which is common workflow in the online RL senarios, stay tuned. We will release this feature in the future release of GEAR.


.. automethod:: SharedDataset.create 

Properties:
------------
.. autoproperty:: SharedDataset.weights

.. autoproperty:: SharedDataset.timesteps

.. autoproperty:: SharedDataset.num_col


Specs:
------------
.. automethod:: SharedDataset.get_column_spec

.. automethod:: SharedDataset.get_column_dtype

.. automethod:: SharedDataset.get_column_shape


Memory Ops:
------------------
.. automethod:: SharedDataset.get_raw_view

.. automethod:: SharedDataset.get_tensor_view

.. automethod:: SharedDataset.set_trajectory

Specials:
------------------
.. automethod:: SharedDataset.__getitem__

.. automethod:: SharedDataset.__len__

.. automethod:: SharedDataset.__setitem__

(De)serializations:
------------------
.. automethod:: SharedDataset.checkpoint

.. automethod:: SharedDataset.load