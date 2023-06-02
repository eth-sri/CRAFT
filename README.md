# Abstract Interpretation of Fixpoint Iterators with Applications to Neural Networks

Here, we provide instructions to reproduce the results of our PLDI'23 paper 'Abstract Interpretation of Fixpoint Iterators with Applications to Neural Networks' and a general overview of the CRAFT codebase.

## Getting started

You can reproduce our results either using the provided docker or in a conda environment.

### Docker
Set up Docker using the `Dockerfile` provided in this repository, e.g.:
``` bash
docker build ./ -t craft 
sudo docker run -it craft
```
To use GPUs with this Docker, please set up Nvidia-Docker (see [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)) and pass the `--gpus all` (or some subset) flag to the `docker run` command.
This docker automatically sets up all files and already runs (`setup.sh`).

### Conda
Alternatively, directly install our tool in a conda environment. 

Please ensure that conda is installed (e.g. https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html).
Then create a conda environment with

```
conda create --name monDEQ_Cert python=3.6.13 -y
```

Activate the environment with 
```
conda activate monDEQ_Cert
```

Install requirements with:
```
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.2 -c pytorch -y
pip install -r requirements.txt
```

### Running Experiments
Please navigate to the directory containing our code:
```
cd lipschitz_mondeqs
```

We provide configuration files for all our experiments in 'configs' which can then be executed using e.g. 
```
python run_experiment.py --path configs/mnist_h40_m20.yaml -n 5
```
This will print results to the console and generate a log file using the following naming scheme based on key parameters
```
[dataset]_[network]_[epsilon]_[pr_alpha]_[fwdbwd_alpha]_[widening]_[joint_projections]_[slope_optimization_level]_[comment]
```
By default, we always evaluate `-n 100` samples, but this can lead to runtimes of over 1 hour per experiment and around 1 week for all experiments, we thus recommend choosing to run on only the first few images.
Note that depreciation warnings are expected and safe to ignore when using our docker or following our installation instructions.

To make the evaluation of all of our ~100 experiments easier, we have collected the corresponding commands in bash scripts corresponding to the core tables and figures of our paper.
Please see the Step-by-Step Instructions for more details on how to run those.
    
## Step-by-Step Instructions
Please find below the instructions to reproduce all results of our experimental evaluation. 

### MONDEQs
Please first activate the right conda environment:
```
conda activate monDEQ_Cert
```

We have prepared the following bash scripts to run all experiments for the main tables and figures of our paper:

```
bash ./run_experiments_table_2.sh
bash ./run_experiments_table_3.sh
bash ./run_experiments_table_4.sh
bash ./run_experiments_figure_12.sh
bash ./run_experiments_HCAS.sh
```

As re-running these experiments requires a substantial amount of time (~1 week), all scripts except `run_experiments_HCAS.sh` can be provided an extra argument, specifying the number of to be evaluated samples. 
By default, we evaluate 100 samples, but recommend a smaller number to test all experiments:
```
bash ./run_experiments_table_2.sh 5
```

To generate Figure 12 from the previously produced logs, use the `monDEQ_plots_fig12.ipynb` notebook.

To parse the HCAS logs and generate Figure 11 from the previously produced logs, use the `CAS_eval.ipynb` notebook.

#### Semi SDP (Table 3)
To use SemiSDP to reproduce our baseline, a Mosek license is required. a free trial or academic license can be obtained [here](https://www.mosek.com/license/request/?i=acp).
Please save the thus obtained license file as `~/mosek/mosek.lic`.

To install SemiSDP, please execute:
```
source setup_semisdp.sh
```

To reproduce our baseline experiments, navigate to the directory containing the (slightly modified) Semi SDP code and activate the right Conda environment:
```
cd semisdp
conda activate SemiSDP
export PATH=$PWD/julia-1.7.2/bin/:$PATH
```

To run all experiments used for comparison in Table 3, run the following script (confirming the download of the dataset on the first execution):
```
bash ./run_semi_comp_experiments.sh
```
Note that these experiments take very long to run (up too 1400s per sample => 1.5 days per setting).

### Application to new Models
To apply CRAFT to new monDEQs, not covered in our paper, you can train them using our training code, e.g. with:
```
python train_models.py -hi 50 -m 20 -pr mnist_short -bs 128 -vs 256 -ds mnist --epoch 10
```

This will train a fully connected monDEQ with a hidden dimension of `-hi` a monotonicity parameter `m` and save it in `CRAFTT/lipschitz_mondeqs/models`. The following naming convention will be used:
```
<provided prefix>_mon_h<hidden dimension>_m<monotonicity parameter>.pt
```
where the prefix is provided with the argument `-pr` in the training command.

To analyse this new models, simply copy one of the existing config files (e.g. `cp configs/mnist_h40_m20.yaml configs/mnist_h50_m20.yaml`) and replace model (`path`) and log (`log`) path as well as the hidden dimension (`hidden`) and monotonicity parameter (`m`). 

You can now run analysis on this new model using the following command:
```
python run_experiment.py --path configs/mnist_h50_m20.yaml -n 20
```


### Numerical Programs
#### Householder's Method
We demonstrate the broader applicability of our method by using it to abstract the iterative numerical Householder method used for square root computation.
Please execute the notebook `Householder_root_iteration.ipynb`.


#### Further applications
To demonstrate how easy it is to apply CRAFT to new problems, beyond what we consider in our paper, we further added Newton's method to the notebook `Householder_root_iteration.ipynb`, which simply required redefining the `step` function.

### Extending this Codebase
Below we give an overview of the most important code components for using the novel CH-Zonotope and our CRAFT verifier of monDEQs. 

#### CH-Zonotope
Our novel CH-Zonotope domain can be found in `ai.new_zonotope.py`. 
For people currently working with HybridZonotope from DiffAI (https://github.com/eth-sri/diffai/blob/master/ai.py), using CH-Zonotope is as easy as simply changing the `domain` argument form `zono` to `chzono` when initializing their abstract elements.
There, you can also find the `can_contain` and `contains` method, which check whether the inner CH-Zonotope may be contained (`outer.can_contain(inner)`) or is guaranteed to be contained (`outer.contains(inner)`) in the outer (proper) CH-Zonotope.

This allows to implement simple analysis methods as demonstrated in the `Householder_root_iteration.ipynb` notebook.

##### Example
Below, we provide a short example of how to use the CH-Zonotope domain (see also `CH_Zonotope_demonstration.ipynb`).

Import the abstract domain and abstract layers:
```
from ai.abstract_layers import Linear
from ai.new_zonotope import HybridZonotope
import torch
```

Now we can create a concrete input `x_concrete` and an CH-Zonotope `x_abstract`, representing an L-Inf cubewith radius 0.1 around it:
```
x_concrete = torch.rand((1,5))
x_abstract = HybridZonotope.construct_from_bounds(x_concrete-0.1, x_concrete+0.1, domain='chzono')
```

To obtain the dimensionwise bounds of the set abstracted by the CH-Zonotope `x_abstract`, we can concretize it:
```
lb, ub = x_abstract.concretize()
```

We can now instantiate a linear layer from `abstract_layers.py` and apply it to both the concrete and abstract input:
```
linear_layer = Linear(in_features=5, out_features=3)
y_concrete = linear_layer(x_concrete)
y_abstract = linear_layer(x_abstract)
```

Similarly, we can define other functions that operate either on concrete or abstract inputs:
```
def add(x, y):
    return x + y

z_concrete = add(x_concrete[:,:3], y_concrete)
z_abstract = add(x_abstract[:,:3], y_abstract)
```

To reduce the number of error terms in our CH-Zonotope, we can now consolidate its errors by calling
```
z_abstract_consolidated, _ = z_abstract.consolidate_errors()
```

We can now scale the resulting CH-Zonotope around `z_concrete` (not its centre):
```
v_abstract = (z_abstract_consolidated - z_concrete) * 0.8 + z_concrete
```

And check if the resulting CH-Zonotope is contained:
```
z_abstract_consolidated.contains(v)
```

#### CRAFT for monDEQs
All experiments on monDEQs use `verify_mondeq.py` to load models, setup logging, and provide problem instances. 
The actual verification as outlined in Algorithm 1 is implemented as an abstract execution (`NormalizedNet.forward_zono`) of the monDEQ in `mondeq_nets.py`.
For more detailed information, we refer to the comments in these two files.
