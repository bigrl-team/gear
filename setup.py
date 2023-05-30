import glob
import os

from setuptools import find_packages, setup
from torch.utils import cpp_extension


class EnvVarError(Exception):
    ...


def find_cuda():
    # TODO: find cuda
    home = os.getenv("CUDA_HOME")
    path = os.getenv("CUDA_PATH")
    if home is not None:
        return home
    elif path is not None:
        return path
    else:
        return "/usr/local/cuda"


def find_nccl():
    home = os.getenv("NCCL_HOME")
    if home is not None:
        return home
    else:
        raise EnvVarError("Set the 'NCCL_HOME' variable before compilation")


def have_cuda():
    return True
    import torch

    return torch.cuda.is_available()


def create_extension(with_cuda=False):
    srcs = []
    srcs += glob.glob("src/*.cc")
    srcs += glob.glob("src/**/*.cc")
    infinity_src_dir = "./third-party/infinity"
    infinity_srcs = [
        infinity_src_dir + "/core/Context.cpp",
        infinity_src_dir + "/memory/Atomic.cpp",
        infinity_src_dir + "/memory/Buffer.cpp",
        infinity_src_dir + "/memory/Region.cpp",
        infinity_src_dir + "/memory/RegionToken.cpp",
        infinity_src_dir + "/memory/RegisteredMemory.cpp",
        infinity_src_dir + "/queues/QueuePair.cpp",
        infinity_src_dir + "/queues/QueuePairFactory.cpp",
        infinity_src_dir + "/requests/RequestToken.cpp",
        infinity_src_dir + "/utils/Address.cpp",
    ]

    srcs += infinity_srcs

    include_dirs = [os.path.abspath("./include/"), os.path.abspath("./third-party/")]
    library_dirs = ["/usr/local/lib64"]
    libraries = ["ibverbs"]
    extra_cxx_flags = [
        "-std=c++17",
        "-fvisibility=hidden",  # pybind requirement
        # TODO: enforce strict build
        # '-Wall',
        # '-Werror',
        # '-Wfatal-errors',
    ]
    if with_cuda:
        cuda_home = find_cuda()
        nccl_home = find_nccl()
        include_dirs += [os.path.join(cuda_home, "include")]
        library_dirs += [os.path.join(cuda_home, "lib64")]
        include_dirs += [os.path.join(nccl_home, "include")]
        library_dirs += [os.path.join(nccl_home, "lib")]
        srcs += glob.glob("src/cuda/*.cu")
        extra_cxx_flags += [
            "-DHAVE_CUDA",
            # "-DGEAR_VERBOSE_DEBUG_ON",
            # "-DINFINITY_DEBUG_ON",
            # "-DINFINITY_ASSERT_ON",
            "-DPYBIND11_DETAILED_ERROR_MESSAGES",
        ]

    return cpp_extension.CppExtension(
        name="libgear",
        sources=srcs,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        with_cuda=with_cuda,
        extra_compile_args={
            "cxx": extra_cxx_flags,
            "nvcc": ["-O3", "--expt-extended-lambda", "-lnuma"],
        },
    )


package_dir = "./"

setup(
    name="gear",
    version="0.2.0",
    author="Hanjing Wang",
    author_email="wanghanjingwhj@sjtu.edu.cn",
    url="https://github.com/VegeWong/PipelineDT",
    description=("GPU-centric data buffer for reinforcement learning"),
    keywords=["PyTorch", "Distributed", "Reinforcement Learning", "Machine Learning"],
    license="Apache",
    python_requires=">=3.6",
    package_dir={
        "": package_dir,
    },
    packages=find_packages(package_dir),
    package_data={
        "libgear": ["*.pyi"],
    },
    ext_modules=[
        create_extension(with_cuda=True),
    ],
    cmdclass={
        # FIXME: parallel build, (pip_install took 1m16s)
        "build_ext": cpp_extension.BuildExtension,
    },
)
