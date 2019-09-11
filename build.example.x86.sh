#!/bin/sh

MPI_HOME=
export PATH=$MPI_HOME/bin:$PATH
export LD_LIBRARY_PATH=$MPI_HOME/lib:$LD_LIBRARY_PATH

CUDA_HOME=

INSTALL_PATH=$PWD/install

./configure --with-cc=$MPI_HOME/bin/mpicc --with-f77=$MPI_HOME/bin/mpif77 --enable-cuda --with-cuda=$CUDA_HOME --prefix=$INSTALL_PATH \
    && echo "if the Configuration Summary looks good; then do 'make shared; make install'"
