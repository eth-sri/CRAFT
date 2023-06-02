#!/bin/bash

pushd . > '/dev/null';
SCRIPT_PATH="${BASH_SOURCE[0]:-$0}";

while [ -h "$SCRIPT_PATH" ];
do
    cd "$( dirname -- "$SCRIPT_PATH"; )";
    SCRIPT_PATH="$( readlink -f -- "$SCRIPT_PATH"; )";
done

cd "$( dirname -- "$SCRIPT_PATH"; )" > '/dev/null';
SCRIPT_PATH="$( pwd; )";
popd  > '/dev/null';

cd
mkdir mosek

cd "$SCRIPT_PATH"
wget https://julialang-s3.julialang.org/bin/linux/x64/1.7/julia-1.7.2-linux-x86_64.tar.gz
tar -xvf julia-1.7.2-linux-x86_64.tar.gz

export PATH=$PWD/julia-1.7.2/bin/:$PATH

conda create --name SemiSDP -y
conda activate SemiSDP

wget https://download.mosek.com/stable/9.3.22/mosektoolslinuxaarch64.tar.bz2
tar -xf mosektoolslinuxaarch64.tar.bz2

julia "$SCRIPT_PATH/semisdp/setup.jl"


