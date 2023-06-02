#!/bin/bash

if test "$1"
then
  N=$1
else
  N=100
fi

ARG="-n $N"


# Only PR iterations
python run_experiment.py --path configs/mnist_matlab_only_PR_02_no_box.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_only_PR_03_no_box.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_only_PR_04_no_box.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_only_PR_05_no_box.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_only_PR_06_no_box.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_only_PR_07_no_box.yaml "$ARG"

python run_experiment.py --path configs/mnist_matlab_only_PR_015.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_only_PR_02.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_only_PR_04.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_only_PR_05.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_only_PR_07.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_only_PR_10.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_only_PR_12.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_only_PR_13.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_only_PR_15.yaml "$ARG"


# Only FwdBwd iterations
python run_experiment.py --path configs/mnist_matlab_FwdBwd_025.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_FwdBwd_027.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_FwdBwd_030.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_FwdBwd_031.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_FwdBwd_032.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_FwdBwd_033.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_FwdBwd_034.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_FwdBwd_035.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_FwdBwd_037.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_FwdBwd_040.yaml "$ARG"

python run_experiment.py --path configs/mnist_matlab_FwdBwd_028_no_box.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_FwdBwd_031_no_box.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_FwdBwd_032_no_box.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_FwdBwd_033_no_box.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_FwdBwd_034_no_box.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_FwdBwd_035_no_box.yaml "$ARG"


# PR then FwdBwd
python run_experiment.py --path configs/mnist_matlab_pr_015.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_pr_02.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_pr_05.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_pr_07.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_pr_10.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_pr_12.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_pr_13.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_pr_14.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_pr_15.yaml "$ARG"

python run_experiment.py --path configs/mnist_matlab_no_box_a015.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_no_box_a02.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_no_box_a03.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_no_box_a04.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_no_box_a05.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_no_box_a06.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_no_box_a07.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_no_box_a10.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_no_box_a12.yaml "$ARG"
python run_experiment.py --path configs/mnist_matlab_no_box_a15.yaml "$ARG"





