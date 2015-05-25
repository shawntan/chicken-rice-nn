import lstm
import numpy as np
import theano
import theano.tensor as T
import theano_toolkit.utils as U
def build(P,character_count,embedding_size=20,hidden_size=50):
	P.V = 0.1 * np.random.randn(character_count,embedding_size)

	lstm_layer_1 = lstm.build(P,
			name = "recurrent_1",
			input_size = embedding_size,
			hidden_size = hidden_size,
#			truncate_gradient = 100
		)

	lstm_layer_2 = lstm.build(P,
			name = "recurrent_2",
			input_size =  hidden_size,
			hidden_size = hidden_size,
#			truncate_gradient = 100
		)

	P.W_output = np.zeros((hidden_size,character_count))
	P.b_output = np.zeros((character_count,))
	
	def predict(X): 							# time x batch_size
		X_rep  = P.V[X] 						# time x batch_size x embedding_size
		_,hiddens_1 = lstm_layer_1(X_rep)
		_,hiddens_2 = lstm_layer_2(hiddens_1)
		hiddens_ = hiddens_2.reshape((hiddens_2.shape[0] * hiddens_2.shape[1], hidden_size))	# time * batch_size x hidden_size
		output = T.nnet.softmax(T.dot(hiddens_,P.W_output) + P.b_output) 
		output = output.reshape((hiddens_2.shape[0] , hiddens_2.shape[1], character_count))	# time x batch_size x output_size
		return output

	return predict
		
def build_sampler(P,character_count,embedding_size=20,hidden_size=50):
	P.V = np.random.randn(character_count,embedding_size)
	P.init_recurrent_1_hidden = 0.5 * np.random.randn(hidden_size)
	P.init_recurrent_1_cell   = 0.5 * np.random.randn(hidden_size)
	P.init_recurrent_2_hidden = 0.5 * np.random.randn(hidden_size)
	P.init_recurrent_2_cell   = 0.5 * np.random.randn(hidden_size)

	lstm_layer_1 = lstm.build_step(P,
			name = "recurrent_1",
			input_size = embedding_size,
			hidden_size = hidden_size
		)

	lstm_layer_2 = lstm.build_step(P,
			name = "recurrent_2",
			input_size =  hidden_size,
			hidden_size = hidden_size
		)
	P.W_output = np.zeros((hidden_size,character_count))
	P.b_output = np.zeros((character_count,))

	def sampler(temp,x,prev_cell_1,prev_hidden_1,prev_cell_2,prev_hidden_2):
		input_embedding = P.V[x]
		cell_1, hidden_1 = lstm_layer_1(input_embedding,prev_cell_1,prev_hidden_1)
		cell_2, hidden_2 = lstm_layer_2(hidden_1,prev_cell_2,prev_hidden_2)
		output = U.vector_softmax(temp*(T.dot(hidden_2,P.W_output) + P.b_output))
		return output,cell_1,hidden_1,cell_2,hidden_2
	return sampler

