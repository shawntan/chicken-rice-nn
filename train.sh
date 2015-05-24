#!/bin/bash

THEANO_FLAGS=device=gpu1 python2 -i -u train.py ~/edmw_data/dataset.txt vocab.pkl > train.log &

tail -f train.log
