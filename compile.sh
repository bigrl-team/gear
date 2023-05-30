#!/bin/bash

rm -r build
mkdir -p build
cd build
export NCCL_HOME=~/nccl/build/
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` -GNinja ..
ninja 2>&1 1>compile.log 

# create bootstrap file
cd ..
rm -f ./build/pipedt.py
touch ./build/pipedt.py
cat << EOF >> ./build/pipedt.py
def __bootstrap__():
    import importlib.util
    import os

    spec = importlib.util.spec_from_file_location(
        __name__,
        os.path.join(
            os.path.split(os.path.abspath(__file__))[0],
            "pipedt.cpython-38-x86_64-linux-gnu.so",
        ),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


__bootstrap__() 
EOF
