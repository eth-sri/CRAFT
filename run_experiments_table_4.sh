#!/bin/bash

if test "$1"
then
  N=$1
else
  N=100
fi

ARG="-n $N"

# Table 4
python run_experiment.py --path configs/mnist_matlab_reference.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_no_zono.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_no_box.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_only_PR.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_FwdBwd_033.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_os0.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_os1.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_contained.yaml "$ARG"
#python run_experiment.py --path configs/mnist_matlab_no_wid.yaml "$ARG"  ### Not included in originally submitted version