FROM ubuntu:18.04

# get essential packages
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends software-properties-common apt-transport-https curl build-essential ca-certificates curl openssl rsync wget cmake g++ gfortran libopenmpi-dev git libgfortran3 libgmp-dev python-dev clang libclang-dev sudo openssh-client gpg-agent python-pip python-setuptools python-configparser python-jinja2 python-mako python-tornado python-zmq libboost-all-dev libfftw3-dev binutils cython libnfft3-dev python-backports-shutil-get-terminal-size zlib1g-dev less nano vim libomp-dev python-backports.functools-lru-cache \
    && \
    apt-get autoremove --purge -y && \
    apt-get autoclean -y && \
    rm -rf /var/cache/apt/* /var/lib/apt/lists/*

ENV CMAKE_PREFIX_PATH=/usr/share/cmake \
CPATH=/usr/include:$CPATH

# MKL 2019.4-070
RUN curl https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB \
    | apt-key add - \
    && echo "deb https://apt.repos.intel.com/mkl all main" > /etc/apt/sources.list.d/intel-mkl.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends intel-mkl-2019.4-070 \
    && rm -rf /var/lib/apt/lists/*

# Set paths for intel libraries, so that we can use them during docker build
ENV INTELPATH           /opt/intel/compilers_and_libraries_2019.4.243/linux
ENV MKLROOT             ${INTELPATH}/mkl
ENV INTEL_LICENSE_FILE  ${INTELPATH}/licenses:/opt/intel/licenses:/root/intel/licenses
ENV LIBRARY_PATH        ${INTELPATH}/tbb/lib/intel64_lin/gcc4.7:${INTELPATH}/compiler/lib/intel64_lin:${INTELPATH}/mkl/lib/intel64_lin:${LIBRARY_PATH}
ENV LD_LIBRARY_PATH     ${INTELPATH}/compiler/lib/intel64:${INTELPATH}/compiler/lib/intel64_lin:${INTELPATH}/tbb/lib/intel64_lin/gcc4.7:${INTELPATH}/compiler/lib/intel64_lin:${INTELPATH}/mkl/lib/intel64_lin:${LD_LIBRARY_PATH}
ENV CPATH               ${INTELPATH}/mkl/include:${CPATH}
ENV NLSPATH             ${INTELPATH}/compiler/lib/intel64/locale/%l_%t/%N:${INTELPATH}/mkl/lib/intel64_lin/locale/%l_%t/%N
ENV PATH                ${INTELPATH}/bin/intel64:${PATH}
ENV PKG_CONFIG_PATH     ${INTELPATH}/mkl/bin/pkgconfig

# Numpy
RUN cd && git clone --branch v1.16.5 https://github.com/numpy/numpy.git numpy
ADD site.cfg /root/numpy/site.cfg
RUN cd /root/numpy/ \
    && python setup.py install \
    && cd .. \
    && rm -rf numpy

# Scipy
RUN cd && git clone --branch v1.2.2  https://github.com/scipy/scipy.git scipy
ADD site.cfg /root/scipy/site.cfg
RUN cd /root/scipy/ \
    && python setup.py install \
    && cd .. \
    && rm -rf scipy

# hdf
RUN cd && wget -q https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.2/src/hdf5-1.10.2.tar.gz \
    && tar xf hdf5-1.10.2.tar.gz \
    && cd hdf5-1.10.2 \
    && ./configure --prefix=/usr --enable-fortran --enable-cxx \
    && make \
    && make install \
    && cd .. \
    && rm -rf hdf5-1.10.2 \
    && rm  hdf5-1.10.2.tar.gz

# h5py
RUN cd && wget -q https://files.pythonhosted.org/packages/74/5d/6f11a5fffc3d8884bb8d6c06abbee0b3d7c8c81bde9819979208ba823a47/h5py-2.8.0.tar.gz \
    && tar xf h5py-2.8.0.tar.gz \
    && cd h5py-2.8.0 \
    && python setup.py install \
    && cd .. \
    && rm -rf h5py-2.8.0 \
    && rm h5py-2.8.0.tar.gz

# set some compiler flags, do not do this earlier! Breaks scipy build
ENV CXX=clang++
ENV CC=clang

# VASP for CSC calculations
ADD csc_vasp.tar.gz /vasp/
RUN  cd /vasp/ \
     && make std \
     && rm -rf src/ build/ arch/

ENV PATH=/vasp/bin:${PATH}

# mpi4py
RUN pip install --no-cache-dir matplotlib mpi4py psutil

# triqs
RUN cd / && mkdir -p triqs && mkdir -p source

ENV CPATH=/triqs/include:${CPATH} \
    PATH=/triqs/bin:${PATH} \
    LIBRARY_PATH=/triqs/lib:${LIBRARY_PATH} \
    LD_LIBRARY_PATH=/triqs/lib:${LD_LIBRARY_PATH} \
    PYTHONPATH=/triqs/lib/python2.7/site-packages:${PYTHONPATH} \
    CMAKE_PREFIX_PATH=/triqs/share/cmake:${CMAKE_PREFIX_PATH}

RUN cd /source && git clone -b 2.2.x https://github.com/TRIQS/triqs triqs.src \
    && mkdir -p triqs.build && cd triqs.build \
    && cmake ../triqs.src -DCMAKE_INSTALL_PREFIX=/triqs -DMPIEXEC_PREFLAGS='--allow-run-as-root' \
    && make -j$(nproc) && make -j$(nproc) test && make install

ENV TRIQS_ROOT=/triqs

# dft_tools
RUN cd /source && git clone -b 2.2.x https://github.com/TRIQS/dft_tools.git dft_tools.src \
    && mkdir -p dft_tools.build && cd dft_tools.build \
    && cmake ../dft_tools.src -DCMAKE_INSTALL_PREFIX=/triqs -DMPIEXEC_PREFLAGS='--allow-run-as-root' \
    && make && make -j$(nproc) test && make install

# cthyb
RUN cd /source && git clone -b 2.2.x https://github.com/TRIQS/cthyb.git cthyb.src \
    && mkdir -p cthyb.build && cd cthyb.build \
    && cmake ../cthyb.src -DCMAKE_INSTALL_PREFIX=/triqs -DMPIEXEC_PREFLAGS='--allow-run-as-root'  \
    && make -j$(nproc) && make -j$(nproc) test && make install

# maxent
RUN cd /source && git clone https://github.com/TRIQS/maxent.git maxent.src \
    && mkdir -p maxent.build && cd maxent.build \
    && cmake ../maxent.src -DCMAKE_INSTALL_PREFIX=/triqs -DMPIEXEC_PREFLAGS='--allow-run-as-root' \
    && make -j$(nproc) && make install

# PyEd Exact diagonalization solver
RUN cd / && git clone https://github.com/HugoStrand/pyed.git
ENV PYTHONPATH=/pyed:$PYTHONPATH

# download solidmft inside container
RUN cd / && git clone https://github.com/materialstheory/soliDMFT.git

# remove source
RUN cd / && rm -rf source

# create a useful work dir
RUN cd / && mkdir work && cd work

# make sure openmp does not start
ENV OMP_NUM_THREADS=1
ENV OMPI_MCA_btl_vader_single_copy_mechanism=none

COPY entrypoint.sh /usr/local/bin/entrypoint.sh


RUN ["chmod", "+x", "/usr/local/bin/entrypoint.sh"]

# change user and group id to match host machine if options are passed accordingly
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
