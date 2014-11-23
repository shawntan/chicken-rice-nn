# coding=utf-8
import theano
import sys
import theano.tensor as T
import numpy         as np
from theano_toolkit import utils as U
import cPickle       as pickle
import random
from theano.printing import Print
from vocab import read_file

def sentences(vocab2id,filename):
	buffer_size = 100
	buffered = [None]*buffer_size
	
	k = 0
	for tokens in read_file(filename):
		if len(tokens) < 5: continue
		#print tokens
		for i in xrange(len(tokens)): tokens[i] = vocab2id.get(tokens[i],-1)
		buffered[k] = tokens
		k = (k+1)%buffer_size
		if k==0:
			random.shuffle(buffered)
			for s in buffered: yield s

