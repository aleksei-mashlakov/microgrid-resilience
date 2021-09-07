# Microgrid resilience

The codes and partly data provided here are used for the experiment simulations described in:

```
@article{2022resilience,
  title={Microgrid resilience enhancement with weather-driven predictive control of battery energy storage},
  author={},
  journal={},
  volume={},
  pages={},
  year={2022},
  publisher={}
}
```

## Installation and running

#### Git installation for Linux machine

    $ mkdir microgrid-resilience
    $ cd microgrid-resilience
    $ git clone https://github.com/aleksei-mashlakov/microgrid-resilience.git .

#### Creation of conda environment

    $ sh ./create_conda.sh

#### Running python scripts

    $ python run_LGBM_solar_prediction.py
    $ python run_CVXPY_optimization.py

#### Removal of conda environment

    $ sh ./remove_conda.sh
