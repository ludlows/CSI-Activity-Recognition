# CSI-Activity-Recognition

Human Activity Recognition using Channel State Information for Wifi Applications

A simple Tensorflow 2.0+ model using Bidirectional LSTM stacked with one Attention Layer.

This code extends the previsous work of paper [A Survey on Behaviour Recognition Using WiFi Channel State Information](http://ieeexplore.ieee.org/document/8067693/) ([corresponding code](https://github.com/ermongroup/Wifi_Activity_Recognition)).

## Dataset Preparation

Download the public dataset from [here](https://drive.google.com/file/d/19uH0_z1MBLtmMLh8L4BlNA0w-XAFKipM/view?usp=sharing).

unzip the Dataset.tar.gz by the following command:

```bash
tar -xzvf Dataset.tar.gz
```

Inside the dataset, there are 7 different human activities: `bed`, `fall`, `pickup`, `run`, `sitdown`, `standup` and `walk`.

## Requirements

Numpy

Tensorflow 2.0+

sklearn

## Performance of the Model with Default Parameters

## Default Parameters

| Parameters for Batching Sequence  |  Value  |  
|-------------------|:-------------:|
| window length     |  1000 |
| Sliding Steps     |  200   |
| Downsample Factor |  2 |
| Activity Present Threshold | 0.6 (60%)|

| Parameters for Deep Learning Model  |  Value  |  
|-------------------|:-------------:|
| # of units in Bidirectional LSTM     |  200 |
| # of units in Attention Hidden State     |  400 |
| Batch Size |  128 |
| Learning Rate | 1e-4|
| Optimizer | Adam |
| # of Epochs | 60 |

## Model Architecture

![Architecture](https://github.com/ludlows/CSI-Activity-Recognition/raw/master/img/model.png)

## Confusion Matrix

![Confusion Matrix](https://github.com/ludlows/CSI-Activity-Recognition/raw/master/img/confusion_matrix.png)

| Label  |  Accuracy  |  
|-------------------|:-------------:|
| bed     |  100% |
| fall    |  97.18%   |
| pickup  |  98.68% |
| run     | 100% |
| sitdown |  95%  |
| standup |   95.56% |
| walk    | 99.51% |

## Usage

Download the code from github.

```bash
git clone https://github.com/ludlows/CSI-Activity-Recognition.git 
```

Enter the code folder.

```bash
cd CSI-Activity-Recognition
```

## Run The Model with Default Parameters

```bash
python csimodel.py your_raw_Dataset_folder
```

Meanwhile, you could also modify the parameters in the `csimodel.py` or change the architectures of neural networks.

This code could be a starting point for your deep learning project using Channel State Information.
