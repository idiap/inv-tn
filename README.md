# Inverse Text Normalization using NMT models


Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/

Written by Pierre-Edouard Honnet <pe[dot]honnet[at]gmail[dot]com>.


This is a bunch of scripts exploiting several tools to perform inverse
text normalization (ITN).  It is based on
[OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) for NMT models,
and [ASRT](https://github.com/idiap/asrt), a text normalization tool
(among other things), used to create the data.  This tool is intended
to be used as an interface between Automatic Speech Recognition (ASR)
and Neural Machine Translation (NMT) modules, as both models generally
used different conventions for the data format.


## Required tools:

1. Moses is used for tokenization (http://www.statmt.org/moses/), but you
can use your own tokenizer if you prefer.

2. [ASRT](https://github.com/idiap/asrt), the Automatic Speech
Recognition Tools, used for text normalization. You can also use
another text normalization tool if you prefer.

3. OpenNMT-py (https://github.com/OpenNMT/OpenNMT-py), to train and
run NMT models. This installation requires a bunch of other things,
but mainly relies on [PyTorch](http://pytorch.org/).

## Using pretrained models

You simply need to adjust the paths in the script `scripts/asr2mt.sh`
($BASEDIR, $MOSESDIR and $NMTDIR, as well as $nmtmodel if you do not
have the same folder structure), then choose which steps to run, and
finally run the script as:

```bash
scripts/asr2mt.sh $inputfile $outputfile
```

Here the input file is supposed to be normalized, or some ASR output.


## Full procedure

An example using Europarl German data, to do ITN on German text (in
our tests we used the German part of the German-French parallel part
of the corpus).

### Useful to set beforehand:
$MOSESBASE
$ASRTBASE
$OPENNMTBASE

### 0) Download the data.

```bash
mkdir -p data/europarl
cd data/europarl
wget http://opus.lingfil.uu.se/download.php?f=Europarl%2Fde-fr.txt.zip && unzip de-fr.txt.zip && rm de-fr.txt.zip Europarl.de-fr.fr
# It was segmented in train / dev /test as:
head -n 1708742 Europarl.de-fr.de > Europarl.train.de-fr.de
tail -n +1708743 Europarl.de-fr.de | head -n 116505 > Europarl.dev.de-fr.de
tail -n +1825248 Europarl.de-fr.de > Europarl.test.de-fr.de
cd ../..
```

### 1) Preprocess the data (Tokenization and "escaping" punctuation we want to keep).

Note that in our tests, we deleted manually some noise in the training
data before this step (e.g. remove the lines with only punctuations,
or redundancies, after sort | uniq, etc.).

```bash
# Tokenize
$MOSESBASE/scripts/tokenizer/tokenizer.perl -l de < data/europarl/Europarl.train.de-fr.de > data/europarl/europarl.train.de.tok.txt
$MOSESBASE/scripts/tokenizer/tokenizer.perl -l de < data/europarl/Europarl.dev.de-fr.de > data/europarl/europarl.dev.de.tok.txt
$MOSESBASE/scripts/tokenizer/tokenizer.perl -l de < data/europarl/Europarl.test.de-fr.de > data/europarl/europarl.test.de.tok.txt

# Escape punctuation
sed -f scripts/replace_punc.sed data/europarl/europarl.train.de.tok.txt > data/europarl/europarl.train.de.tok.punc.txt
sed -f scripts/replace_punc.sed data/europarl/europarl.dev.de.tok.txt > data/europarl/europarl.dev.de.tok.punc.txt
sed -f scripts/replace_punc.sed data/europarl/europarl.test.de.tok.txt > data/europarl/europarl.test.de.tok.punc.txt
```


### 2) Create parallel data with ASRT (and putting back punctuation).

```bash
export NLTK_DATA=$adjust_to_your_environment # based on your asrt install
export PYTHONPATH=$ASRTBASE/local/lib/python2.7/site-packages # or based on your asrt install
mkdir -p data/europarl_normalized

$ASRTBASE/data-preparation/python/run_data_preparation.py -i data/europarl/europarl.train.de.tok.punc.txt -l 2 -r $ASRTBASE/examples/resources/regex.csv -s -m -o data/europarl_normalized
sed -f scripts/replace_back_punc.sed data/europarl_normalized/sentences_german.txt > data/europarl_normalized/europarl.train.de.tok.punc.norm.txt

$ASRTBASE/data-preparation/python/run_data_preparation.py -i data/europarl/europarl.dev.de.tok.punc.txt -l 2 -r $ASRTBASE/examples/resources/regex.csv -s -m -o data/europarl_normalized
sed -f scripts/replace_back_punc.sed data/europarl_normalized/sentences_german.txt > data/europarl_normalized/europarl.dev.de.tok.punc.norm.txt

$ASRTBASE/data-preparation/python/run_data_preparation.py -i data/europarl/europarl.test.de.tok.punc.txt -l 2 -r $ASRTBASE/examples/resources/regex.csv -s -m -o data/europarl_normalized
sed -f scripts/replace_back_punc.sed data/europarl_normalized/sentences_german.txt > data/europarl_normalized/europarl.test.de.tok.punc.norm.txt

```

### 3) Prepare data to train the model.

```bash
python $OPENNMTBASE/preprocess.py -train_src data/europarl_normalized/europarl.train.de.tok.punc.norm.txt \
       -train_tgt data/europarl/europarl.train.de.tok.txt \
       -valid_src data/europarl_normalized/europarl.dev.de.tok.punc.norm.txt \
       -valid_tgt data/europarl/europarl.dev.de.tok.txt \
       -src_vocab_size 80000 -tgt_vocab_size 80000 \
       -save_data data/Europarl_punc.atok
```

### 4) Train a system.

```bash
mkdir -p asr2mt-models-punc
python $OPENNMTBASE/train.py -data data/Europarl_punc.atok.train.pt \
-save_model asr2mt-models-punc/asr2mt_model -gpus 0

```

### 5) To test the system, see the "Using pretrained models" section.


## Release Notes

The models tested were trained using punctuation in the normalized
text.

Some other models have been trained using no punctuation (it means,
not using the 2 steps with `sed` in step 1 and step 2).  This means
that the model will try to recover punctuation during "translation".
In practice, it should be better to do it with punctuation in both
normalized and not normalized versions (if we assume that ASR is
followed by or has a punctuation prediction module).


