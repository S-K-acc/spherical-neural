# Spherical metadensity functional learning

This repository contains code, data and neural models for the methods presented in:

**Spherical metadensity functional learning for inhomogeneous classical fluids**  
*Stefanie M. Kampa, Florian Sammüller and Matthias Schmidt*


## Instructions

### Setup

A recent version of [Julia](https://julialang.org/downloads/) needs to be installed on your system (Julia 1.12 was used for development).
Launch the Julia interpreter within this directory and type `]` to enter the package manager.
Activate the environment and install the required packages as follows:

```julia
activate .
instantiate
```

Type backspace to exit the package manager.
Start a Jupyter server:

```julia
using IJulia
jupyterlab()
```

### Usage

#### Training the metadensity functional

The directory `Train_meta` contains data and code for training the required neural metadensity functionals from scratch, (see the README in this directory for further instructions.)

#### Application of the metadensity functional

In the directory `Application_meta`, we demonstrate how to utilize the neural metadensity functional for the prediction of density profiles, two-body direct correlation functionals and inversion of pair structure. 

#### Functional for hard spheres 

The directory `HS` contains data and code for training a neural functional for hard spheres and code for applying this functional. 
