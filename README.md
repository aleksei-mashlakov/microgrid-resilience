# Microgrid resilience

The codes and partly data provided here are used for the experiment simulations described in:

```
@article{gutierrez2021weather,
  title={Weather-Driven Predictive Control of a Battery Storage for Improved Microgrid Resilience},
  author={Gutierrez-Rojas, Daniel and Mashlakov, Aleksei and Brester, Christina and Niska, Harri and Kolehmainen, Mikko and Narayanan, Arun and Honkapuro, Samuli and Nardelli, Pedro HJ},
  journal={IEEE Access},
  volume={9},
  pages={163108-163121},
  year={2021}
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
