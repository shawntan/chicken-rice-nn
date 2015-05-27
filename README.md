# theano-nlp

This repository is named theano-nlp, but it isn't exactly what it says on the tin.
I've implemented a character-level language model here. I *could* change the name
of the repository, but I've decided against it. I think this would be useful in
my NLP efforts in the future so I'm keeping the name.

### How to use it

1. Get a dataset. It just needs to be a simple one full of lines of ASCII text.
2. Generate the vocabulary file:

        python vocab.py data_file.txt vocab.pkl

3. Train the model. This step will take the longest, and may need several
   adjustments to get right:

        python train.py data_file.txt vocab.pkl model.pkl

4. Sample from your trained model

        python sample.py vocab.pkl model.pkl

### HTML page

I've also included a little page that runs the model in your browser. Here are
the instructions to "deploy" your own model in such a way.

1. Turn the .pkl model file you have into a .json file with the following
   incantation:

        python -c "import pickle,json; print json.dumps({ k:w.tolist() for k,w in pickle.load(open('model.pkl')).iteritems() })" > params.json

2. Copy it into the html directory and host that thing somewhere. If you want
   to test it before you host it,

        python -mSimpleHTTPServer

   inside the `html/` directory.
