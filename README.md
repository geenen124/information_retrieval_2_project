# pointer summarizer

Pytorch implementation of *[Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)*
Base version reference: *[pointer_summerizer](https://github.com/atulkum/pointer_summarizer)*

* Requires python 3.x and pytorch 0.4

# Data:
* Follow data generation instruction from https://github.com/abisee/cnn-dailymail 

# How to Install PyRouge
--------------------------------------------
1. Clone old PyRouge Repository into separate directory `git clone https://github.com/andersjo/pyrouge.git pyrouge_ancient`
2. Clone new PyRouge Repository `git clone https://github.com/bheinzerling/pyrouge.git`
3. Run `pip install pyrouge`
4. Set the Rouge-1.5.5 Installation Directory (N.B. This needs to be an absolute path!!!)
  `pyrouge_set_rouge_path <absolute_path_of_current_directory>/pyrouge_ancient/tools/ROUGE-1.5.5/`
5. Run `cd pyrouge`
6. Check to see if it worked by running `python -m pyrouge.test`

# How to install python pyrouge
--------------------------------------------
1. Follow the install instructions of `git clone https://github.com/pltrdy/rouge`


# How to run training:
--------------------------------------------
* You might need to change some path and parameters in data_util/config.py
* For training run start_train.sh, for decoding run start_decode.sh, and for evaluating run run_eval.sh
OR
```
export PYTHONPATH=$PYTHONPATH:'pwd'
python training_ptr_gen/train.py
```

# How to load a model checkpoint for training/evaluation/decoding
* **Note:** For training passing the initial checkpoint is optional argument, but for evaluation and decoding passing the checkpoint is must.
* For training:
```
python training_ptr_gen/train.py -m <absolute/path/to/model/checkpoint>
```
* For evaluation and decoding:
```
python training_ptr_gen/eval.py <absolute/path/to/model/checkpoint>
python training_ptr_gen/decode.py <absolute/path/to/model/checkpoint>
```
