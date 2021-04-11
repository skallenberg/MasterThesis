# Multigrid in Neural Networks
Analysis of CNNs for Image Classification to evaluate the impact of Multigrid Methods on performance.

To run the code set the parameters in the file config.toml accordingly and comment all model types in grid_run.py that should NOT be executed.
Afterwards the results can be run with

<code>python grid_run.py

This creates output in form of tensorboard events that can be viewed through

<code>tensorboard --logdir=./data/models/logs/runs

Alternatively paramater counts can be produced by running

<code>python parameter_count.py
