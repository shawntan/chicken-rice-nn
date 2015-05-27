# theano-nlp

This repository is named theano-nlp, but it isn't exactly what it says on the tin.
I've implemented a character-level language model here. I *could* change the name
of the repository, but I've decided against it. I think this would be useful in
my NLP efforts in the future either way.

## How to use it

1. Get a dataset. It just needs to be a simple one full of lines of ASCII text.
2. Generate the vocabulary file:
```bash
python vocab.py data_file.txt vocab.pkl
```
3. Train the model. This step will take the longest, and may need several
   adjustments to get right:
```bash
python train.py data_file.txt vocab.pkl model.pkl
```
4. Sample from your trained model
```bash
python sample.py vocab.pkl model.pkl
```
