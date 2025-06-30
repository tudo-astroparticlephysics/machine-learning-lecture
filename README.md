# A machine learning lecture <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a> [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tudo-astroparticlephysics/machine-learning-lecture/main)

This collection of notebooks was started for a lecture on machine learning at the Universitat Autònoma de Barcelona.
It has since grown into a large part of the statistical methods lecture (SMD) at the Physics department at TU Dortmund University.
It contains some mathematical derivations and small excersises to play with.

As of now, you need to execute this notebook within the project folder since it imports some plotting functions from the `ml` module.


![TU Logo](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Technische_Universit%C3%A4t_Dortmund_Logo.svg/800px-Technische_Universit%C3%A4t_Dortmund_Logo.svg.png)

## License

The programming code examples in this material are shared under the GnuGPLv3 license.
The lecture material (e.g. jupyter notebooks) are shared under the Creative Commons Attribution-NonCommercial License: https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt, so they cannot be used for commercial training / tutorials / lectures.


## Lectures

1. Data-Preprocessing and feature selection (smd_pca.ipynb)
2. Introduction to supervised machine learning (smd_ml.ipynb, part 1)
3. Validation, Bias-Variance-Tradeoff, ensemble methods (smd_ml.ipynb, part 2)
4. Unsupervised learning, clustering (smd_unsupervised.ipynb)
5. Example on FACT Data and Boosting (smd_fact_boosting.ipynb)
6. Neural Networks (smd_neural_networks.ipynb)


# Running the notebooks


## Install `conda`

To make sure, all needed packages are installed in an environment for these lectures, we use
`conda`.

Download and install [Anaconda](https://www.anaconda.com/products/individual#Downloads) for a large collection of packages or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for a minimal starting point.

## Setup the environment

After installing conda, run

```
$ conda env create -f environment.yml
```

This will create a new conda environment with all needed packages for these lectures
named `ml`.

To use this environment, run
```
$ conda activate ml
```
everytime before you start working on these lectures.

Before committing any changes, make sure the pre-commit hooks are installed:
```
$ pre-commit install
```

From time to time, we will update the `environment.yml` with new versions or
additional packages, to then update your environment, run:
```
$ conda env update -f environment.yml
```


## Running the notebooks

Just run

```
$ jupyter notebook
```
this will open your default browser at the overview page, where you can select each of
the notebooks.
