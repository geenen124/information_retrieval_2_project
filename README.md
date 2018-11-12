# pointer summarizer

Pytorch implementation of *[Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)*
Base version reference: *[pointer_summerizer](https://github.com/atulkum/pointer_summarizer)*

* Requires python 3.x and pytorch 0.4

# Data:
* Follow data generation instruction from https://github.com/abisee/cnn-dailymail 

# pyrogue
```
git clone https://github.com/andersjo/pyrouge
cd pyrouge
python setup.py install
```

# How to run training:
--------------------------------------------
* You might need to change some path and parameters in data_util/config.py
* For training run start_train.sh, for decoding run start_decode.sh, and for evaluating run run_eval.sh
OR
```
export PYTHONPATH=$PYTHONPATH:'pwd'
python training_ptr_gen/train.py
```


# How to load a model checkpoint for training/testing
* Pass the absolute path to the checkpoint as argument. 
* For ex:
```
python training_ptr_gen/train.py -m <absolute/path/to/model/checkpoint>
```
