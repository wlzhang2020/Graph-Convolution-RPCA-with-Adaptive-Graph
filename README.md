# Graph Convolution RPCA with Adaptive Neighbors

This repository is the official implementation of GRPCA. 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Preprocess the data
To better evaluate the performance of various methods, all the datasets are processed by following step:
1) Normalize the data.
2) Pollute the data. Sample 20% data points randomly and 20% features of them are reset by random values.

The raw data can be found in `./data`, the processed data are in `./data_occ`


## Training and evaluation

To train and evaluate the model(s) in the paper, set the parameters in run.py, and run this command:

```train
python run.py
```