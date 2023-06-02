#!/bin/bash

if test "$1"
then
  N=$1
else
  N=100
fi

ARG="-n $N"

# Table 2
python run_experiment.py --path configs/mnist_h40_m20.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_reference.yaml "$ARG"
python run_experiment.py --path configs/mnist_h100_m20.yaml "$ARG"
python run_experiment.py --path configs/mnist_h200_m20.yaml "$ARG"
python run_experiment.py --path configs/mnist_conv_small_m20.yaml "$ARG"

python run_experiment.py --path configs/cifar_h200_m20.yaml "$ARG"
python run_experiment.py --path configs/cifar_conv_small_m20_jp.yaml "$ARG"

