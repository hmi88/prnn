# Performance RNN: Generating Music with Expressive Timing and Dynamics

Pytorch implementation of [Performance RNN](https://magenta.tensorflow.org/performance-rnn)



## 0. Requirments

- Pretty_midi
- numpy
- pytorch >= 1.0



## 1. Usage

```
# Data Tree
config.data_dir
└── config.data_name/
        └── midi/
        │     └── *.mid
        ├── event/ # processed
        │     └── *.data/
        └── note/  # processed 
                └── *.data/
  
# Project Tree
PRNN
├── PRNN_src/
│       ├── data/
│       │     └── *.py
│       ├── loss/
│       │     └── *.py
│       ├── model/
│       │     └── *.py
│       └── *.py
└── PRNN_exp/
         ├── log/
         ├── model/
         └── save/         

```



### 1.1  Train

```
# Note based
python train.py --data_type "note"

# Event based
python train.py --data_type "event"
```



### 1.2 Test

```
# Note based
python train.py --is_train false --data_type "note"

# Event based
python train.py --is_train false --data_type "Event"
```



