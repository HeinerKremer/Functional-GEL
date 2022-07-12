# Functional-GEL
Implementation of Functional Generalized Empirical Likelihood Estimators 
for conditional moment restriction problems and code to reproduce the experiments in the corresponding 
[paper](https://proceedings.mlr.press/v162/kremer22a.html).

Parts of the implementation are based on the codebase for the [Variational Method of Moments](https://github.com/CausalML/VMM) estimator.

## Installation
To install the package, create a virtual environment and run the setup file from within the folder containing this README, e.g. using the following commands:
```bash
python3 -m venv fgel-venv
source fgel-venv/bin/activate
pip install -e .
```

## Syntax
The syntax used to train FGEL estimators is described below. Applying the method to a problem at hand requires phrasing the problem
as a class inheriting from the [AbstractExperiment](experiments/abstract_experiment.py) class, 
refer to [exp_heteroskedastic.py](experiments/exp_heteroskedastic.py) for an example.
```python
from fgel.kernel_fgel import KernelFGEL
from experiments.exp_heteroskedastic import HeteroskedasticNoiseExperiment

# Initialize model and data
exp = HeteroskedasticNoiseExperiment(theta=1.7)
exp.setup_data(n_train=200, n_val=200, n_test=20000)
model = exp.init_model()

# Train FGEL estimator
estimator = KernelFGEL(model=model, reg_param=1e-7)
estimator.train(x_train=exp.x_train, z_train=exp.z_train, x_val=exp.x_val, z_val=exp.z_val)

# Make prediction
y_pred = model(exp.x_test[0])
```

[comment]: <> (## Reproducibility)

[comment]: <> (The experimental results presented in the [paper]&#40;https://proceedings.mlr.press/v162/kremer22a.html&#41; can be reproduced by running the script [run_experiment.py]&#40;run_experiment.py&#41; via)

[comment]: <> (```)

[comment]: <> (python3 run_experiment.py --experiment exp --run_all --method method --rollouts 50)

[comment]: <> (```)

[comment]: <> (with `exp in ['heteroskedastic', 'network_iv']` and `methods in []`.)

## Citation
If you use parts of the code in this repository for your own research purposes, please consider citing:
```
@InProceedings{pmlr-v162-kremer22a,
  title = 	 {Functional Generalized Empirical Likelihood Estimation for Conditional Moment Restrictions},
  author =       {Kremer, Heiner and Zhu, Jia-Jie and Muandet, Krikamol and Sch{\"o}lkopf, Bernhard},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {11665--11682},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v162/kremer22a/kremer22a.pdf},
  url = 	 {https://proceedings.mlr.press/v162/kremer22a.html},
  abstract = {Important problems in causal inference, economics, and, more generally, robust machine learning can be expressed as conditional moment restrictions, but estimation becomes challenging as it requires solving a continuum of unconditional moment restrictions. Previous works addressed this problem by extending the generalized method of moments (GMM) to continuum moment restrictions. In contrast, generalized empirical likelihood (GEL) provides a more general framework and has been shown to enjoy favorable small-sample properties compared to GMM-based estimators. To benefit from recent developments in machine learning, we provide a functional reformulation of GEL in which arbitrary models can be leveraged. Motivated by a dual formulation of the resulting infinite dimensional optimization problem, we devise a practical method and explore its asymptotic properties. Finally, we provide kernel- and neural network-based implementations of the estimator, which achieve state-of-the-art empirical performance on two conditional moment restriction problems.}
}
```
