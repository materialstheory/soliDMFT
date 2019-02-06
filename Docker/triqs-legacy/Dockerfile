FROM materialstheory/base-bionic-gnu

# need cython
RUN pip install cython

# Numpy
RUN cd && git clone --branch v1.13.3 https://github.com/numpy/numpy.git numpy
ADD site.cfg /root/numpy/site.cfg
RUN cd /root/numpy/ \
    && python setup.py install \
    && cd .. \
    && rm -rf numpy

# Scipy
RUN cd && git clone --branch v0.19.1  https://github.com/scipy/scipy.git scipy
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
RUN cd && wget -q https://files.pythonhosted.org/packages/41/7a/6048de44c62fc5e618178ef9888850c3773a9e4be249e5e673ebce0402ff/h5py-2.7.1.tar.gz \
    && tar xf h5py-2.7.1.tar.gz \
    && cd h5py-2.7.1 \
    && python setup.py install \
    && cd .. \
    && rm -rf h5py-2.7.1 \
    && rm h5py-2.7.1.tar.gz

# needed python packages that dont need manual installation
RUN pip install --no-cache-dir matplotlib mpi4py

# set some compiler flags
ENV CFLAGS="-m64 -O3 -Wl,--no-as-needed"
ENV CXXFLAGS="-m64 -O3 -Wl,--no-as-needed"
ENV LDFLAGS="-ldl -lm"
ENV FFLAGS="-m64 -O3"

# triqs
RUN cd / && mkdir -p triqs && mkdir -p source

RUN cd /source && git clone -b 1.4.2 https://github.com/TRIQS/triqs triqs.src \
    && mkdir -p triqs.build && cd triqs.build \
    && cmake ../triqs.src -DCMAKE_INSTALL_PREFIX=/triqs -DLAPACK_LIBS=/opt/intel/compilers_and_libraries_2018.3.222/linux/mkl/lib/intel64_lin/libmkl_rt.so \
    && make -j$(nproc) &&  make install

ENV TRIQS_ROOT=/triqs

ENV CPATH=/triqs/include:${CPATH} \
    PATH=/triqs/bin:${PATH} \
    LIBRARY_PATH=/triqs/lib:${LIBRARY_PATH} \
    LD_LIBRARY_PATH=/triqs/lib:${LD_LIBRARY_PATH} \
    PYTHONPATH=/triqs/lib/python2.7/site-packages:${PYTHONPATH} \
    CMAKE_PREFIX_PATH=/triqs/share/cmake:${CMAKE_PREFIX_PATH}

# dft_tools
RUN cd /source && git clone -b vasp https://github.com/TRIQS/dft_tools.git dft_tools.src \
    && mkdir -p dft_tools.build && cd dft_tools.build \
    && cmake ../dft_tools.src -DTRIQS_PATH=/triqs \
    && make -j$(nproc) && make install

# cthyb
RUN cd /source && git clone -b 1.4.2 https://github.com/TRIQS/cthyb.git cthyb.src \
    && mkdir -p cthyb.build && cd cthyb.build \
    && cmake ../cthyb.src -DTRIQS_PATH=/triqs \
    && make -j$(nproc) && make install

# maxent
RUN cd /source && git clone https://github.com/TRIQS/maxent.git maxent.src \
    && mkdir -p maxent.build && cd maxent.build \
    && cmake ../maxent.src -DTRIQS_PATH=/triqs \
    && make -j$(nproc) && make install

# VASP for CSC calculations
ADD csc_vasp.tar.gz /vasp/
RUN  cd /vasp/ \
     && make std \
     && rm -rf src/ build/ arch/

ENV PATH=/vasp/bin:${PATH}

# remove source
RUN cd / && rm -rf source

# create a useful work dir
RUN cd / && mkdir work && cd work

# make sure openmp does not start
ENV OMP_NUM_THREADS=1

COPY entrypoint.sh /usr/local/bin/entrypoint.sh

RUN ["chmod", "+x", "/usr/local/bin/entrypoint.sh"]

# change user and group id to match host machine if options are passed accordingly
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
