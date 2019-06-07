FROM materialstheory/base-bionic-sci-python

# set some compiler flags
ENV CFLAGS="-m64 -O3 -Wl,--no-as-needed"
ENV CXXFLAGS="-m64 -O3 -Wl,--no-as-needed"
ENV LDFLAGS="-ldl -lm"
ENV FFLAGS="-m64 -O3"

# triqs
RUN cd / && mkdir -p triqs && mkdir -p source

RUN cd /source && git clone https://github.com/TRIQS/cpp2py cpp2py.src \
    && mkdir -p cpp2py.build && cd cpp2py.build \
    && cmake ../cpp2py.src -DCMAKE_INSTALL_PREFIX=/triqs  -DPYTHON_LIBRARY=/usr/lib/python2.7/config-x86_64-linux-gnu/libpython2.7.so \
    && make && make install

ENV CPATH=/triqs/include:${CPATH} \
    PATH=/triqs/bin:${PATH} \
    LIBRARY_PATH=/triqs/lib:${LIBRARY_PATH} \
    LD_LIBRARY_PATH=/triqs/lib:${LD_LIBRARY_PATH} \
    PYTHONPATH=/triqs/lib/python2.7/site-packages:${PYTHONPATH} \
    CMAKE_PREFIX_PATH=/triqs/share/cmake:${CMAKE_PREFIX_PATH}

RUN cd /source && git clone -b 2.1.0 https://github.com/TRIQS/triqs triqs.src \
    && mkdir -p triqs.build && cd triqs.build \
    && cmake ../triqs.src -DCMAKE_INSTALL_PREFIX=/triqs -DLAPACK_LIBS=/opt/intel/compilers_and_libraries_2018.3.222/linux/mkl/lib/intel64_lin/libmkl_rt.so \
    && make -j$(nproc) && make -j$(nproc) test && make install

ENV TRIQS_ROOT=/triqs

# dft_tools
RUN cd /source && git clone -b 2.1.0 https://github.com/TRIQS/dft_tools.git dft_tools.src \
    && mkdir -p dft_tools.build && cd dft_tools.build \
    && cmake ../dft_tools.src -DCMAKE_INSTALL_PREFIX=/triqs \
    && make && make -j$(nproc) test && make install

# cthyb
RUN cd /source && git clone -b 2.1.0 https://github.com/TRIQS/cthyb.git cthyb.src \
    && mkdir -p cthyb.build && cd cthyb.build \
    && cmake ../cthyb.src -DCMAKE_INSTALL_PREFIX=/triqs \
    && make -j$(nproc) && make -j$(nproc) test && make install

# maxent
RUN cd /source && git clone https://github.com/TRIQS/maxent.git maxent.src \
    && mkdir -p maxent.build && cd maxent.build \
    && cmake ../maxent.src -DCMAKE_INSTALL_PREFIX=/triqs \
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

### the following part is not necessary for code execution. All for jupyter lab 
# make sure that jupyter lab is installed
RUN pip install --no-cache-dir jupyter ipywidgets jupyterlab nodejs pandas shapely descartes ipympl pymatgen==2018.12.12

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends texlive dvipng
    texlive-latex-extra texlive-fonts-recommended nodejs npm \
    && \
    apt-get autoremove --purge -y && \
    apt-get autoclean -y && \
    rm -rf /var/cache/apt/* /var/lib/apt/lists/*

# jupyter lab extension
RUN jupyter-labextension install @jupyter-widgets/jupyterlab-manager \
    && jupyter-labextension install jupyter-matplotlib \
    && jupyter-labextension install @jupyterlab/toc
RUN jupyter-lab build

# expose port for jupyter
EXPOSE 8378 
### end jupyter lab part

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
COPY jupyter_start.sh /usr/local/bin/jupyter.sh

RUN ["chmod", "+x", "/usr/local/bin/entrypoint.sh"]
RUN ["chmod", "+x", "/usr/local/bin/jupyter.sh"]

# change user and group id to match host machine if options are passed accordingly
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
