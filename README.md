## Description
This project is an implementation of Roth and Lapatas "Neural Semantic Role Labeling with Dependency Path Embeddings"[1], a Semantic Role Labeling Model with a LSTM at the core.

## Requirements
#### Software
Python >= 3.4.3 until this (https://github.com/tensorflow/tensorflow/issues/4588) will be in tensorflow, then 3.X should be fine. <br />
tensorflow >= 0.10.0 <br />
Perl >= 5.8.1 for evaluation with the CoNNL 2009 Scorer<br />

#### Data
Download the CoNLL 2008 [2] or 2009 [3] Shared Task data:<br />
2008: https://catalog.ldc.upenn.edu/LDC2009T12<br />
2009: https://catalog.ldc.upenn.edu/LDC2012T04<br />

#### Scorer
If you want to evaluate a dataset with the official CoNLL Scorer you have to download it here:<br />
2009: https://ufal.mff.cuni.cz/conll2009-st/scorer.html<br />
Put the scorer named 'eval09.pl' into the 'score_scripts' folder. The python script will use it internally and can even evaluate 2008 format input files. The scorer is executed by a 'perl' command, so make sure to have it installed. You can switch of warnings in the perl script for better readability.

#### Model
I trained a model on the CoNLL2008 'train.closed' file. The parameters where set like in the paper, except there where only 2 iterations for each argument identification model and 5 iterations for argument classification. This model scores 73.89% Labeled F1 score on the in-domain test file and 67.61% on the out-of-domain test file.<br />
Download model: https://www.dropbox.com/sh/au325tntluwe8us/AAC8tUeO5SH6txKUT5DC2Mu6a?dl=0 <br />
After the download place all *.model and *.meta files into the 'model' folder. Only the main *.model file (the one without LSTM in it) needs to be passed to the run script, the others will be identified by their names.


## Run
#### Training
    python3 run.py train ./data/train/train.closed conll2008 ./output/
Trains a model with the 'train.closed' file from the CoNLL-2008 Shared Task. Writes the model files into 'output'. Be sure to set your desired parameters in 'config.cfg'. <br />
The provided model took around 21 hours to train on a p2.xlarge EC2 instance with one NVIDIA K80 GPU. 

#### Testing
    python3 run.py test ./data/test.wsj/test.wsj.closed.GOLD conll2008 ./output/ ./model/train.closed_pi_ai_ac.model
The input files needs to have gold labels to be evaluated on. Writes the predicted labels to 'output/[input_file].PRED' and the gold labels to 'output/[input_file].GOLD'. The full evaluation is written to 'output/[input_file].RESULTS'.

#### Prediction
    python3 run.py predict ./data/test.wsj/my_own_sentences.conll conll2009 ./output/ ./model/train.closed_pi_ai_ac.model
The input file format has to be the CoNLL 2008 or 2009 format (without argument labels, but with sense disambiguated predicates!). Predicts the argument labels and writes them to 'output/[input_file].PRED'.

##ToDos
1. Implement Predicate Prediction and Disambiguation
2. Implement Reranker

<br />
<br />

[1] Roth and Lapata, 2016, https://arxiv.org/abs/1605.07515 <br />
[2] Surdeanu et al., 2008, http://dl.acm.org/citation.cfm?id=1596411 <br />
[3] Hajic et al., 2009, http://dl.acm.org/citation.cfm?id=1596324.1596352


