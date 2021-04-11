# Multigrid in Neural Networks
Analysis of CNNs for Image Classification to evaluate the impact of Multigrid Methods on performance.

To run the code set the parameters in the file config.toml accordingly and comment all model types in grid_run.py that should NOT be executed.
Afterwards the results can be run with

```python grid_run.py```
  
This creates output in form of tensorboard events that can be viewed through

```tensorboard --logdir=./data/models/logs/runs```
  
Alternatively paramater counts can be produced by running

```python parameter_count.py```

All required packages can be installed via pip through

```pip install -r requirements.txt``` 

PyTorch may have to be installed manually as of 11.04.2021 the install via pip results in an error. PyTorch can be installed manually through 

```pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html```
