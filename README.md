# Neural-Tree-Search
Neural Tree Search for active slam

## acme-more-mcts

## Installation
fork [acme-more-mcts](https://github.com/bwfbowen/acme-more-mcts) and run following command from the main directory(where `setup.py` is located):
```sh
pip install .[jax,tf,testing,envs]
```


Habitat is also needed:
```sh
pip install habitat-sim
pip install habitat-api
```

## Setup
The project requires datasets in a `data` folder in the following format (same as habitat-api):
```
Neural-Tree-Search/
  data/
    scene_datasets/
      gibson/
        Adrian.glb
        Adrian.navmesh
        ...
    datasets/
      pointnav/
        gibson/
          v1/
            train/
            val/
            ...
```
Please download the data using the instructions here: https://github.com/facebookresearch/habitat-api#data

## Getting started
To run the code:
```python
python run_acme.py
```

## Results

![dis](https://github.com/bwfbowen/Neural-Tree-Search/assets/104526323/5bec0425-e6e9-401a-a391-6904bb9aac2b)
