import theano
import theano.tensor as T
import numpy as np
import cPickle as pickle

from itertools import izip

from theano_toolkit import utils as U
from theano_toolkit import updates
from theano_toolkit.parameters import Parameters

def build(P,name,input_size,hidden_size,truncate_gradient=-1):
	name_init_hidden = "init_%s_hidden"%name
	name_init_cell   = "init_%s_cell"%name
	P[name_init_hidden] = np.zeros((hidden_size,))
	P[name_init_cell]   = np.zeros((hidden_size,))

	_step, transform = _build_step(P,name,input_size,hidden_size,batched=True)

	def lstm_layer(X):
		init_hidden = T.tanh(P[name_init_hidden])
		init_cell   = P[name_init_cell]
		init_hidden_batch = T.alloc(init_hidden,X.shape[1],hidden_size)
		init_cell_batch   = T.alloc(init_cell,X.shape[1],hidden_size)
		[cell,hidden],_ = theano.scan(
				_step,
				sequences    = [transform(X)],
				outputs_info = [init_cell_batch,init_hidden_batch],
				truncate_gradient=truncate_gradient
			)
		return cell,hidden
	return lstm_layer

def build_step(P,name,input_size,hidden_size):
	_step,_ = _build_step(P,name,input_size,hidden_size,batched=False)
	def step(x,prev_cell,prev_hid):
		x = x.dimshuffle('x',0)
		prev_cell = prev_cell.dimshuffle('x',0)
		prev_hid  = prev_hid.dimshuffle('x',0)
		cell, hid = _step(x,prev_cell,prev_hid)
		return cell[0], hid[0]
	return step


def _build_step(P,name,input_size,hidden_size,batched):
	name_W_input  = "W_%s_input"%name
	name_W_hidden = "W_%s_hidden"%name
	name_W_cell   = "W_%s_cell"%name
	name_b        = "b_%s"%name
	P[name_W_input]  = 0.08 * np.random.rand(input_size,  hidden_size*4)
	P[name_W_hidden] = 0.08 * np.random.rand(hidden_size, hidden_size*4)
	P[name_W_cell]   = 0.08 * np.random.rand(hidden_size, hidden_size*3)
	bias_init = np.zeros((4,hidden_size),dtype=np.float32)
	bias_init[1] = 2.5
	P[name_b] = bias_init

	V_if = P[name_W_cell][:,0*hidden_size:2*hidden_size]
	V_o  = P[name_W_cell][:,2*hidden_size:3*hidden_size]

	biases = P[name_b]
	b_i = biases[0]
	b_f = biases[1]
	b_c = biases[2]
	b_o = biases[3]


	def _transform(x):
		"""
		Input dimensions:  time x batch_size x input_size
		Output dimensions: time x batch_size x 4 x hidden_size
		"""
		x_flatten = x.reshape((x.shape[0] * x.shape[1],input_size)) # time * batch_size x input_size
		_transformed_x = T.dot(x_flatten,P[name_W_input])			# time * batch_size x 4 * hidden_size
		_transformed_x = _transformed_x\
							.reshape((x.shape[0],x.shape[1],4,hidden_size))
																	# time x batch_size x 4 * hidden_size
		return _transformed_x

	def _step(x,prev_cell,prev_hid):
		if batched:
			transformed_x = x
		else:
			transformed_x = T.dot(x,P[name_W_input]).reshape((1,4,hidden_size))

		# batch_size x hidden_size

		batch_size = transformed_x.shape[0]
		transformed_x    = transformed_x.reshape((batch_size,4,hidden_size))						# batch_size x 4 x hidden_size
		transformed_hid  = T.dot(prev_hid,P[name_W_hidden]).reshape((batch_size,4,hidden_size)) # batch_size x 4 x hidden_size
		transformed_cell = T.dot(prev_cell,V_if).reshape((batch_size,2,hidden_size))			# batch_size x 2 x hidden_size


		transformed_x_ = transformed_x.dimshuffle(1,0,2)
		x_i = transformed_x_[0]
		x_f = transformed_x_[1]
		x_c = transformed_x_[2]
		x_o = transformed_x_[3]   # batch_size x hidden_size

		transformed_hid_ = transformed_hid.dimshuffle(1,0,2)
		h_i = transformed_hid_[0]
		h_f = transformed_hid_[1]
		h_c = transformed_hid_[2]
		h_o = transformed_hid_[3] # batch_size x hidden_size
		
		transformed_cell_ = transformed_cell.dimshuffle(1,0,2)
		c_i = transformed_cell_[0]
		c_f = transformed_cell_[1] # batch_size x hidden_size

		in_lin     = x_i + h_i + b_i + c_i
		forget_lin = x_f + h_f + b_f + c_f
		cell_lin   = x_c + h_c + b_c

		in_gate      = T.nnet.sigmoid(in_lin)
		forget_gate  = T.nnet.sigmoid(forget_lin)
		cell_updates = T.tanh(cell_lin)
		in_gate.name = "in_gate"
		forget_gate.name = "forget_gate"
		cell_updates.name = "cell_updates"

		cell = forget_gate * prev_cell + in_gate * cell_updates

		out_lin = x_o + h_o + b_o + T.dot(cell,V_o)
		out_gate = T.nnet.sigmoid(out_lin)
		out_gate.name = "out_gate"

		hid = out_gate * T.tanh(cell)

		return cell,hid

	return _step,_transform


if __name__ == "__main__":
	P = Parameters()
	X = T.imatrix('X')									# batch_size x time
	P.V = np.zeros((8,8),dtype=np.int32)

	
	P.W_output = np.zeros((15,8),dtype=np.int32)
	lstm_layer = build(P,
			name = "test",
			input_size = 8,
			hidden_size =15 
		)

	X_rep = P.V[X]														# batch_size x time x embedding_size
	X_rep_ = X_rep.dimshuffle(1,0,2)									# time x batch_size x embedding_size
	_,hidden = lstm_layer(X_rep_)										# time x batch_size x hidden_size
	hidden_ = hidden.reshape((hidden.shape[0] * hidden.shape[1], 15))	# time * batch_size x hidden_size
	output = T.nnet.softmax(T.dot(hidden_,P.W_output))					# time * batch_size x output_size
	output = output.reshape((hidden.shape[0] , hidden.shape[1],8))		# time x batch_size x output_size

	delay = 5
	label = X.T[:-delay]			# time - delay x batch_size
	predicted = output[delay:]		# time - delay x batch_size x output_size

	cost = -T.mean(T.sum(
			T.log(predicted[
				T.arange(label.shape[0]).dimshuffle(0,'x'),
				T.arange(label.shape[1]).dimshuffle('x',0),
				label
			]), axis=0), axis=0)

	params = P.values()
	gradients = T.grad(cost,wrt=params)


	update_methods = {
#			'standard': [ (p, p - 0.001 * g) for p,g in zip(params,gradients) ],
			'rmsprop' : updates.rmsprop(params,gradients),
#			'adadelta': updates.rmsprop(params,gradients),
		}
	P.save('init.pkl')
	for update_method in update_methods:
		print "Using update method:",update_method
		with open('train.%s.smart_init.log'%update_method,'w') as log:

			train = theano.function(
					inputs = [X],
					outputs = cost,
					updates = update_methods[update_method],
				)

			P.load('init.pkl')

			while True:
				cost_val = train(np.random.randint(0,8,size=(10,20)).astype(np.int32))
				print cost_val
				if cost_val < 0.01:
					break
		P.save('lstm.%s.smart_init.pkl'%update_method)

			


	




