import sys
import numpy as np
import theano
import theano.tensor as T
import numpy as np
import cPickle as pickle
import vocab
import model
from theano_toolkit.parameters import Parameters
from theano_toolkit import updates

import data_io
import itertools
def make_batch_train(cost_fun,params):
	batch = T.imatrix('batch')
	[costs,disp_costs],_ = theano.map(cost_fun,sequences=batch)
	batch_cost = T.mean(costs)

	print "Calculating gradient..."
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
		batch_array = np.ones((len(batch),max_length),dtype=np.int32)
		for i,l in enumerate(batch):
			batch_array[i,:len(l)] = l
		return _train(batch_array)
	print "Done."
	return train

def make_functions(inputs,outputs,params,grads,lr):
	shapes = [ p.get_value().shape for p in params ]
	acc_grads = [ theano.shared(np.zeros(s,dtype=np.float32)) for s in shapes ]
	count = theano.shared(np.float32(0))
	acc_update = [ (a,a+g) for a,g in zip(acc_grads,grads) ] + [ (count,count + 1) ]

#	deltas = acc_grads
	deltas      = [ ag / count for ag in acc_grads ]
	grads_norms = [ T.sqrt(T.sum(g**2)) for g in deltas ]
	deltas      = [ T.switch(T.gt(n,5),5*g/n,g)
					for n,g in zip(grads_norms,deltas) ]
	
#	param_update = [ (p, p - lr * g) for p,g in zip(params,deltas) ]
	param_update = updates.rmsprop(params,deltas)
	
	clear_update = [ 
			(a,np.zeros(s,dtype=np.float32)) 
			for a,s in zip(acc_grads,shapes) 
		] + [ (count,0) ]
	acc = theano.function(
			inputs  = inputs,
			outputs = outputs,
			updates = acc_update,
			on_unused_input='warn'
		)
	update = theano.function(
			inputs=[lr],
			updates = param_update + clear_update,
			on_unused_input='warn'
		)
	return acc,update




if __name__ == "__main__":
	filename = sys.argv[1]
	vocab_file = sys.argv[2]
	char2id = vocab.load(vocab_file)
	P = Parameters()
	lang_model = model.build(P,
			character_count = len(char2id) + 1,
			embedding_size = 20,
			hidden_size = 50
		)

	def cost(X):
		char_prob_dist = lang_model(X[:-1])
		char_prob = char_prob_dist[T.arange(X.shape[0]-1),X[1:]]
		return -T.sum(T.log(char_prob)),\
				2**(-T.mean(T.log2(char_prob)))
					


	params = P.values()
	train = make_batch_train(cost,params)

	data_stream = data_io.stream(filename,char2id)
	data_stream = data_io.sortify(data_stream,key=lambda x: len(x))
	batch_data_stream = data_io.batch(data_stream)
	batch_data_stream = data_io.randomise(batch_data_stream)

	learning_rate = 0.001
	seen = 0
	while True:
		for batch in batch_data_stream:
			avg_cost = train(batch)
			print avg_cost
			seen += len(batch)
			if seen%1000==0:
				P.save('tmp.model.pkl')
		P.save('model.pkl')

