import sys
import cPickle as pickle
import numpy as np
import theano
import theano.tensor as T

from theano_toolkit.parameters import Parameters
from theano_toolkit import updates

import vocab
import model
import data_io

import itertools
def make_batch_train(P,cost,end_id):
	batch = T.imatrix('batch')
	costs,disp_costs = cost(batch,P)
	batch_cost = T.mean(costs)

	print "Calculating gradient..."
	params = P.values()
	grads = T.grad(batch_cost,wrt=params)
	grads_norms = [ T.sqrt(T.sum(g**2)) for g in grads ]
	deltas      = [ T.switch(T.gt(n,5),5*g/n,g)
					for n,g in zip(grads_norms,grads) ]
	


	print "Compiling function..."
	_train = theano.function(
			inputs = [batch],
			outputs = T.mean(disp_costs),
			updates = updates.rmsprop(params,deltas)
		)
	def train(batch):
		max_length = max(len(l) for l in batch)
		batch_array = end_id * np.ones((len(batch),max_length),dtype=np.int32)
		for i,l in enumerate(batch):
			batch_array[i,:len(l)] = l
		return _train(batch_array)
	print "Done."
	return train




if __name__ == "__main__":
	filename = sys.argv[1]
	vocab_file = sys.argv[2]
	char2id = vocab.load(vocab_file)
	P = Parameters()
	lang_model = model.build(P,
			character_count = len(char2id) + 1,
			embedding_size = 100,
			hidden_size = 100
		)

	def cost(X,P): # batch_size x time
		eps = 1e-3
		X = X.T 										# time x batch_size
		char_prob_dist = lang_model(X[:-1])				# time x batch_size x output_size
		char_prob_dist = (1 - 2 * eps) * char_prob_dist + eps
		label_prob = char_prob_dist[
				T.arange(X.shape[0]-1).dimshuffle(0,'x'),
				T.arange(X.shape[1]).dimshuffle('x',0),
				X[1:]
			]												# time x batch_size
		cross_entropy = -T.sum(T.log(label_prob),axis=0)
		display_cost = 2**(-T.mean(T.log2(label_prob),axis=0))
		l2 = sum(T.sum(p**2) for p in P.values())

		return cross_entropy, display_cost
				
					
	params = P.values()
	train = make_batch_train(P,cost,end_id=char2id["\n"])
	seen = 0
	batch_size = 32
	epoch = 1
	from pprint import pprint
	id2char = pickle.load(open(vocab_file))
	print id2char
	while True:
		print "Epoch:",epoch
		print "Batch size:",batch_size
		data_stream = data_io.stream(filename,char2id)
		data_stream = ( s for s in data_stream if len(s) <= 500 )
		data_stream = data_io.randomise(data_stream,buffer_size=1024)
		data_stream = data_io.sortify(data_stream,key=lambda x:len(x),buffer_size=512)
		batch_data_stream = data_io.batch(data_stream,batch_size=batch_size)
		batch_data_stream = data_io.randomise(batch_data_stream)
		for batch in batch_data_stream:
			avg_cost = train(batch)
			if np.isnan(avg_cost):
				pprint([''.join(id2char[c] for c in l[1:]) for l in batch])
				exit(1)
			print avg_cost
			seen += len(batch)
			if seen > 1000:
				print "Saving..."
				P.save('tmp.model.pkl')
				seen = 0

		P.save('model.pkl')
		epoch += 1

