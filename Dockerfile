### Base image
#FROM rapidsai/rapidsai-dev:22.10-cuda11.5-devel-ubuntu20.04-py3.9 as base
FROM rapidsai/rapidsai-dev:23.02-cuda11.8-devel-ubuntu22.04-py3.10 as base
RUN mv /opt/conda/envs/rapids/include/boost/mp11 /opt/conda/envs/rapids/include/boost/mp11_do_not_use

### CLion remote builder
FROM base AS clion-remote-builder
RUN <<EOF cat > /root/tmp_bashrc && mv /root/tmp_bashrc /root/.bashrc
export PATH='/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
export NVARCH="$NVARCH"
export NVIDIA_REQUIRE_CUDA="$NVIDIA_REQUIRE_CUDA"
export NV_CUDA_CUDART_VERSION="$NV_CUDA_CUDART_VERSION"
export NV_CUDA_COMPAT_PACKAGE="$NV_CUDA_COMPAT_PACKAGE"
export CUDA_VERSION="$CUDA_VERSION"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
export NVIDIA_VISIBLE_DEVICES="$NVIDIA_VISIBLE_DEVICES"
export NVIDIA_DRIVER_CAPABILITIES="$NVIDIA_DRIVER_CAPABILITIES"
export NV_CUDA_LIB_VERSION="$NV_CUDA_LIB_VERSION"
export NV_NVTX_VERSION="$NV_NVTX_VERSION"
export NV_LIBNPP_VERSION="$NV_LIBNPP_VERSION"
export NV_LIBNPP_PACKAGE="$NV_LIBNPP_PACKAGE"
export NV_LIBCUSPARSE_VERSION="$NV_LIBCUSPARSE_VERSION"
export NV_LIBCUBLAS_PACKAGE_NAME="$NV_LIBCUBLAS_PACKAGE_NAME"
export NV_LIBCUBLAS_VERSION="$NV_LIBCUBLAS_VERSION"
export NV_LIBCUBLAS_PACKAGE="$NV_LIBCUBLAS_PACKAGE"
export NV_LIBNCCL_PACKAGE_NAME="$NV_LIBNCCL_PACKAGE_NAME"
export NV_LIBNCCL_PACKAGE_VERSION="$NV_LIBNCCL_PACKAGE_VERSION"
export NCCL_VERSION="$NCCL_VERSION"
export NV_LIBNCCL_PACKAGE="$NV_LIBNCCL_PACKAGE"
export NV_CUDA_CUDART_DEV_VERSION="$NV_CUDA_CUDART_DEV_VERSION"
export NV_NVML_DEV_VERSION="$NV_NVML_DEV_VERSION"
export NV_LIBCUSPARSE_DEV_VERSION="$NV_LIBCUSPARSE_DEV_VERSION"
export NV_LIBNPP_DEV_VERSION="$NV_LIBNPP_DEV_VERSION"
export NV_LIBNPP_DEV_PACKAGE="$NV_LIBNPP_DEV_PACKAGE"
export NV_LIBCUBLAS_DEV_VERSION="$NV_LIBCUBLAS_DEV_VERSION"
export NV_LIBCUBLAS_DEV_PACKAGE_NAME="$NV_LIBCUBLAS_DEV_PACKAGE_NAME"
export NV_LIBCUBLAS_DEV_PACKAGE="$NV_LIBCUBLAS_DEV_PACKAGE"
export NV_LIBNCCL_DEV_PACKAGE_NAME="$NV_LIBNCCL_DEV_PACKAGE_NAME"
export NV_LIBNCCL_DEV_PACKAGE_VERSION="$NV_LIBNCCL_DEV_PACKAGE_VERSION"
export NV_LIBNCCL_DEV_PACKAGE="$NV_LIBNCCL_DEV_PACKAGE"
export LIBRARY_PATH="$LIBRARY_PATH"
export CONDA_DIR="/opt/conda"
export LANG="en_US.UTF-8"
export LC_ALL="en_US.UTF-8"
export LANGUAGE="en_US:en"
export DEBIAN_FRONTEND="$DEBIAN_FRONTEND"
export CC="/usr/bin/gcc"
export CXX="/usr/bin/g++"
export CUDAHOSTCXX="/usr/bin/g++"
export CUDA_HOME="$CUDA_HOME"
export CONDARC="$CONDARC"
export RAPIDS_DIR="$RAPIDS_DIR"
export DASK_LABEXTENSION__FACTORY__MODULE="$DASK_LABEXTENSION__FACTORY__MODULE"
export DASK_LABEXTENSION__FACTORY__CLASS="$DASK_LABEXTENSION__FACTORY__CLASS"
export NCCL_ROOT="$NCCL_ROOT"
export PARALLEL_LEVEL="$PARALLEL_LEVEL"
export CUDAToolkit_ROOT="$CUDAToolkit_ROOT"
export CUDACXX="$CUDACXX"
`cat /root/.bashrc`
EOF
RUN ( \
    echo 'LogLevel DEBUG2'; \
    echo 'PermitRootLogin yes'; \
    echo 'PasswordAuthentication yes'; \
    echo 'Subsystem sftp /usr/lib/openssh/sftp-server'; \
  ) > /etc/ssh/sshd_config_test_clion \
  && mkdir /run/sshd
RUN yes password | passwd root
# https://ubuntuforums.org/archive/index.php/t-914151.html
# https://stackoverflow.com/questions/69259311/why-do-i-got-conda-command-not-found-when-building-a-docker-while-in-base-im
SHELL ["/bin/bash", "-i", "-c"]
RUN apt-get update && apt-get install -y ssh gdb rsync || mkdir -p /var/run/dbus && dpkg --configure -a && echo "Configured" && apt-get install -y ssh gdb rsync
CMD ["/usr/sbin/sshd", "-D", "-e", "-f", "/etc/ssh/sshd_config_test_clion"]

### Build meta parser
FROM base AS parser-build

RUN mkdir -p /opt/meta-json-parser
WORKDIR /opt/meta-json-parser

COPY third_parties third_parties/
COPY benchmark benchmark/
COPY test test/
COPY meta_cudf meta_cudf/
COPY include include/
COPY CMakeLists.txt CMakeLists.txt

RUN mkdir build
WORKDIR build

RUN /opt/conda/envs/rapids/bin/cmake -DUSE_LIBCUDF=1 -DLOCAL_LIB=1 ..
RUN make -j

### Build python binding
FROM parser-build AS python-binding

#COPY libmeta-cudf-parser-1.a libmeta-cudf-parser-1.a # for debug
ENV LD_LIBRARY_PATH="/opt/meta-json-parser/build:${LD_LIBRARY_PATH}"
COPY python_binding python_binding/
COPY meta_cudf/parser.cuh python_binding/
WORKDIR python_binding
RUN make -j
WORKDIR ..
