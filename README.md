# theano-nlp

This repository is named theano-nlp, but it isn't exactly what it says on the tin.
I've implemented a character-level language model here. I *could* change the name
of the repository, but I've decided against it. I think this would be useful in
my NLP efforts in the future either way.

## How to use it

1. Get a dataset. It just needs to be a simple one full of lines of ASCII text.
2. Generate the vocabulary file:

        python vocab.py data_file.txt vocab.pkl

3. Train the model. This step will take the longest, and may need several
   adjustments to get right:

        python train.py data_file.txt vocab.pkl model.pkl

4. Sample from your trained model

        python sample.py vocab.pkl model.pkl

## HTML page

I've also included a little page that runs the model in your browser. Here are
the instructions to "deploy" your own model in such a way.
