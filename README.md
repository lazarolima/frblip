# FRBlip

Code for FRB Mock simulations. This is a beta version only available only for BINGO members. Please do not use it for any other goals or share the code with someone outside the collaboration.

## Installation

First clone the repository:

```
>>> git clone https://github.com/lazarolima/frblip.git
```

Create and activate a conda environment with the specific python version:

```
>>> conda create --name env_frblip python=3.10
>>> conda activate env_frblip
```

Within the env_frblip environment, install the pyccl and pygedm dependencies as follows:

```
>>> conda install -c conda-forge pyccl=2.3.0
>>> conda install -c conda-forge pygedm
```

To install the remaining dependencies and frblip:

```
>>> cd frblip
>>> pip install -r requirements.txt
>>> pip install -e .
```
