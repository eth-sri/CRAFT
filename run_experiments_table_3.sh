#!/bin/bash

if test "$1"
then
  N=$1
else
  N=100
fi

ARG="-n $N"

# Table 3
python run_experiment.py --path configs/mnist_h40_m20_eps_01.yaml "$ARG"
python run_experiment.py --path configs/mnist_h40_m20_eps_02.yaml "$ARG"
python run_experiment.py --path configs/mnist_h40_m20.yaml "$ARG"
python run_experiment.py --path configs/mnist_h40_m20_eps_07.yaml "$ARG"
python run_experiment.py --path configs/mnist_h40_m20_eps_1.yaml "$ARG"

python run_experiment.py --path configs/mnist_matlab_eps_01.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_eps_02.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_reference.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_eps_07.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_eps_1.yaml "$ARG"

