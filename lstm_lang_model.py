import theano
import sys
import theano.tensor as T
import numpy         as np
from theano_toolkit import utils as U
from theano_toolkit import updates
from theano_toolkit.parameters import Parameters
import cPickle       as pickle
from theano.printing import Print
from vocab import read_file

def create_vocab_vectors(P,vocab2id,size):
	return U.initial_weights(len(vocab2id) + 1,size)

def word_cost(probs,Y):
	lbl_probs = probs[T.arange(Y.shape[0]),Y]
	return -T.sum(T.log(lbl_probs)), -T.mean(T.log2(lbl_probs))

def build_lstm_step(P,word_vector_size,hidden_state_size):
	P.W_input_in  = U.initial_weights(word_vector_size,hidden_state_size)
	P.W_hidden_in = U.initial_weights(hidden_state_size,hidden_state_size)
	P.W_cell_in   = U.initial_weights(hidden_state_size,hidden_state_size)
	P.b_in        = U.initial_weights(hidden_state_size)

	P.W_input_forget  = U.initial_weights(word_vector_size,hidden_state_size)
	P.W_hidden_forget = U.initial_weights(hidden_state_size,hidden_state_size)
	P.W_cell_forget   = U.initial_weights(hidden_state_size,hidden_state_size)
	P.b_forget        = U.initial_weights(hidden_state_size)

	P.W_input_output  = U.initial_weights(word_vector_size,hidden_state_size)
	P.W_hidden_output = U.initial_weights(hidden_state_size,hidden_state_size)
	P.W_cell_output   = U.initial_weights(hidden_state_size,hidden_state_size)
	P.b_output        = U.initial_weights(hidden_state_size)

	P.W_input_cell  = U.initial_weights(word_vector_size,hidden_state_size)
	P.W_hidden_cell = U.initial_weights(hidden_state_size,hidden_state_size)
	P.b_cell        = U.initial_weights(hidden_state_size)

	P.init_h = U.initial_weights(hidden_state_size)
	P.init_c = U.initial_weights(hidden_state_size)

	def step(x,prev_h,prev_c):
		input_gate = T.nnet.sigmoid(
				T.dot(x,P.W_input_in) +\
				T.dot(prev_h,P.W_hidden_in) +\
				T.dot(prev_c,P.W_cell_in) +\
				P.b_in
			)

		forget_gate = T.nnet.sigmoid(
				T.dot(x,P.W_input_forget) +\
				T.dot(prev_h,P.W_hidden_forget) +\
				T.dot(prev_c,P.W_cell_forget) +\
				P.b_forget
			)

		curr_c = forget_gate * prev_c + input_gate * T.tanh(
				T.dot(x,P.W_input_cell) +\
				T.dot(prev_h,P.W_hidden_cell) +\
				P.b_cell
			)

		output_gate = T.nnet.sigmoid(
				T.dot(x,P.W_input_output) +\
				T.dot(prev_h,P.W_hidden_output) +\
				T.dot(curr_c,P.W_cell_output) +\
				P.b_output
			)
		
		curr_h = output_gate * T.tanh(curr_c)

		return curr_h,curr_c

	return step







def create_model(ids,vocab2id,size):
	word_vector_size  = size
	hidden_state_size = size
	
	P = Parameters()
	P.V = create_vocab_vectors(P,vocab2id,word_vector_size)
	P.W_predict = np.zeros(P.V.get_value().shape).T
	P.b_predict = np.zeros((P.V.get_value().shape[0],))
	X = P.V[ids]

	step = build_lstm_step(P,word_vector_size,hidden_state_size)

	[states,_],_ = theano.scan(
			step,
			sequences    = [X],
			outputs_info = [P.init_h,P.init_c]
		)

	scores = T.dot(states,P.W_predict) + P.b_predict
	scores = T.nnet.softmax(scores)

	log_likelihood, cross_ent = word_cost(scores[:-1],ids[1:])
	cost = log_likelihood #+ 1e-4 * sum( T.sum(abs(w)) for w in P.values() )
	obv_cost = cross_ent
	return scores, cost, obv_cost, P

def make_accumulate_update(inputs,outputs,parameters,gradients,update_method=updates.adadelta):
	acc = [ U.create_shared(np.zeros(p.get_value().shape)) for p in parameters ]
	count = U.create_shared(np.int32(0))
	acc_update = [ (a,a + g) for a,g in zip(acc,gradients) ] + [ (count,count+1) ]
	acc_gradient = theano.function(
				inputs = inputs,
				outputs = outputs,
				updates = acc_update
			)
	avg_gradient = [ a/count for a in acc ]
	clear_update = [ (a,0.*a) for a,g in zip(acc,parameters) ] + [ (count,0) ]
	train_acc = theano.function(
			inputs=[],
			updates=update_method(parameters,avg_gradient) + clear_update
		)
	return acc_gradient,train_acc


def training_model(vocab2id,size):
	ids = T.ivector('ids')
	scores, cost, obv_cost, P = create_model(ids,vocab2id,size)
	
	parameters = P.values()
	gradients = T.grad(cost,wrt=parameters)
	print "Computed gradients"
	acc_gradient,train_acc = make_accumulate_update(
			inputs  = [ids],
			outputs = obv_cost,
			parameters = parameters, gradients=gradients,
			update_method=updates.adadelta
		)
	test = theano.function(
			inputs  = [ids],
			outputs = obv_cost
		)

	predict = theano.function(
			inputs  = [ids],
			outputs = T.argmax(scores,axis=1)
		)

	return predict,acc_gradient,train_acc,test,P

def run_test(vocab2id,test_file,test):
	total,count = 0,0
	for s in sentences(vocab2id,test_file):
		s = np.array(s,dtype=np.int32)
		score = test(s)
		length = len(s) - 1
		total += score * length
		count += length
	return total/count

if __name__ == "__main__":
	from train import sentences

	vocab_file = sys.argv[1]
	sentence_file = sys.argv[2]
	test_file = sys.argv[3]

	vocab2id = pickle.load(open(vocab_file,'r'))
	id2vocab = [None]*len(vocab2id)
	for k,v in vocab2id.iteritems(): id2vocab[v]=k

	predict,acc_gradient,train_acc,test,P = training_model(vocab2id,20)

#	import os.path
#	if os.path.isfile('params'): 
#		print "Loading params..."
#		P.load('params')
	print "Starting training..."
	max_test = np.inf
	for epoch in range(10):
		count = 0
		for s in sentences(vocab2id,sentence_file):
			s = np.array(s,dtype=np.int32)
			score = acc_gradient(s)
			count += 1
			if count%50 == 0:
				train_acc()
				print score
		test_score = run_test(vocab2id,test_file,test)
		print "Epoch %d, Test result: %0.4f"%(epoch,test_score)
		if test_score < max_test:
			max_test = test_score
			P.save('params')
		else:
			print "Final:",max_test
			exit()
