#!/bin/env bash

pacman --noconfirm -Sy
pacman --noconfirm -Su
pacman --noconfirm -Sy
pacman --noconfirm -Su

echo "#########################################################################"
echo "### Maybe you need to restart msys after update of system components! ###"
echo "#########################################################################"

pacman -S --needed --noconfirm \
        make \
        tar \
        git \
        subversion \
        mercurial \
        unzip \
        wget \
        diffutils \
        patch

# ordinary cmake fails for win32 detection in mysys
pacman -r --noconfirm cmake

pacman -S --needed --noconfirm --overwrite \
        mingw-w64-x86_64-cmake \
        mingw-w64-x86_64-gcc \
        mingw-w64-x86_64-gcc-fortran \
        mingw-w64-x86_64-openblas \
        mingw-w64-x86_64-suitesparse \
        mingw-w64-x86_64-cppunit \
        mingw-w64-x86_64-clang 
        
pacman -S --needed --noconfirm --overwrite \
        mingw-w64-x86_64-doxygen \
        mingw-w64-x86_64-llvm 

# we need to replace these hard coded paths from the LLVM cmake config
sed -i 's/C:\/repo\/mingw-w64-clang\/src\/build-x86_64/C:\/msys64\/mingw64/'  /mingw64/share/llvm/cmake/LLVMConfig.cmake
sed -i 's/C:\/repo\/mingw-w64-clang\/src\/llvm-3.7.0.src\/cmake\/modules/C:\/msys64\/mingw64\/share\/llvm\/cmake/' /mingw64/share/llvm/cmake/LLVMConfig.cmake

sed -i 's/FATAL_ERROR "The imported target/WARNING "The imported target/' /mingw64/share/llvm/cmake/LLVMExports.cmake
