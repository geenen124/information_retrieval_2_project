# Global and Local Critical Policy Learning for Abstractive Summarization


N.B.
Includes a Pytorch implementation of *[Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)*
Base version reference: *[pointer_summerizer](https://github.com/atulkum/pointer_summarizer)*

# Training
* To train an initial Seq-2-Seq Network run the `training_ptr_gen/train_pytorch.py` file
* To train the Network using the Policy Gradient run the `seqGAN/main.py` file (be sure to point to your pretrained Seq2Seq model if you want good results). The default reward level is the global sentence level reward. 
* Pass `--word-level` to train with local word level rewards
* Pass `--combined` to train with the combined rewards
* Adjust the `data_util/config.py` as necessary to adjust paths and training settings
* N.B. You may need to run `export PYTHONPATH=$PYTHONPATH:'pwd'` in order to properly configure the references


# Data:
* Follow data generation instruction from https://github.com/abisee/cnn-dailymail 

# Dependencies
* Requires python 3.x and pytorch 0.4
* To install the dependencies run `pip install -r requirements.txt`
* In order to evaluate results using the ROUGE metric please follow the instructions below (a working PERL installation is required)


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


# Evaluation of saved models
* **Note:** For training passing the initial checkpoint is optional argument, but for evaluation and decoding passing the checkpoint is must.
* For training:
```
python seqGAN/main.py -m <absolute/path/to/model/checkpoint>
```
* For evaluation and decoding:
```
python training_ptr_gen/eval.py <absolute/path/to/model/checkpoint>
python training_ptr_gen/decode.py <absolute/path/to/model/checkpoint>
```
