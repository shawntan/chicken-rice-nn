import lstm
import numpy as np
import theano
import theano.tensor as T
import theano_toolkit.utils as U
def build(P,character_count,embedding_size=20,hidden_size=50):
	P.V = np.random.randn(character_count,embedding_size)
	lstm_layer_1 = lstm.build(P,
			name = "recurrent_1",
			input_size = embedding_size,
			hidden_size = hidden_size
		)
	lstm_layer_2 = lstm.build(P,
			name = "recurrent_2",
			input_size =  hidden_size,
			hidden_size = hidden_size
		)


	P.W_output = np.zeros((hidden_size,character_count))
	P.b_output = np.zeros((character_count,))
	
	def predict(X):
		input_embeddings = P.V[X]
		_,hiddens_1 = lstm_layer_1(input_embeddings)
		_,hiddens_2 = lstm_layer_2(hiddens_1)
		output = T.nnet.softmax(T.dot(hiddens_2,P.W_output) + P.b_output) 
		return output
	return predict
		
def build_sampler(P,start_id,end_id,character_count,embedding_size=20,hidden_size=50):
	from theano.tensor.shared_randomstreams import RandomStreams
	theano_rng = RandomStreams(np.random.randint(2 ** 30))
	P.V = np.random.randn(character_count,embedding_size)

	P.init_recurrent_1_hidden = 0.5 * np.random.randn(hidden_size)
	P.init_recurrent_1_cell   = 0.5 * np.random.randn(hidden_size)
	lstm_layer_1 = lstm.build_step(P,
			name = "recurrent_1",
			input_size = embedding_size,
			hidden_size = hidden_size
		)

	P.init_recurrent_2_hidden = 0.5 * np.random.randn(hidden_size)
	P.init_recurrent_2_cell   = 0.5 * np.random.randn(hidden_size)
	lstm_layer_2 = lstm.build_step(P,
			name = "recurrent_2",
			input_size =  hidden_size,
			hidden_size = hidden_size
		)
	P.W_output = np.zeros((hidden_size,character_count))
	P.b_output = np.zeros((character_count,))

	def sample(temp):
		def _sample(x,prev_cell_1,prev_hidden_1,prev_cell_2,prev_hidden_2):
			input_embedding = P.V[x]
			cell_1, hidden_1 = lstm_layer_1(input_embedding,prev_cell_1,prev_hidden_1)
			cell_2, hidden_2 = lstm_layer_2(hidden_1,prev_cell_2,prev_hidden_2)
			output = U.vector_softmax(temp*(T.dot(hidden_2,P.W_output) + P.b_output))
			c = T.cast(T.argmax(theano_rng.multinomial(pvals=output)),'int8')
#			c = T.cast(T.argmax(output),'int8')
			stop = T.eq(c,end_id)
			return [c,cell_1,hidden_1,cell_2,hidden_2,]#, {}, \
#					theano.scan_module.until(stop)
		sequences,updates = theano.scan(
				_sample,
				outputs_info = [
					start_id,
					P.init_recurrent_1_cell,
					T.tanh(P.init_recurrent_1_hidden),
					P.init_recurrent_2_cell,
					T.tanh(P.init_recurrent_2_hidden),
				],
				n_steps = 100
			)
	
		return sequences[0],updates
	return sample

