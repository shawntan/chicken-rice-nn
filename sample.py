import sys
import numpy as np
import cPickle as pickle

import theano
import theano.tensor as T
import numpy as np

import vocab
import model
from theano_toolkit.parameters import Parameters

if __name__ == "__main__":
	vocab_file = sys.argv[1]
	model_file = sys.argv[2]
	temp_input = np.float32(sys.argv[3])
	if len(sys.argv) > 4:
		prime_str = sys.argv[4]
	else:
		prime_str = ""
	char2id = vocab.load(vocab_file)
	id2char = pickle.load(open(vocab_file))
	P = Parameters()
	sampler = model.build_sampler(P,
			character_count = len(char2id) + 1,
			embedding_size = 20,
			hidden_size = 100
		)
	P.load(model_file)
	temp = T.scalar('temp')
	char = T.iscalar('char')
	p_cell_1,p_hidden_1,p_cell_2,p_hidden_2 = T.vector("p_cell_1"),T.vector("p_hidden_2"),T.vector("p_cell_2"),T.vector("p_hidden_2")

	output,cell_1,hidden_1,cell_2,hidden_2 = sampler(temp,char,p_cell_1,p_hidden_1,p_cell_2,p_hidden_2)
	sample = theano.function(
			inputs=[temp,char,p_cell_1,p_hidden_1,p_cell_2,p_hidden_2],
			outputs=[output,cell_1,hidden_1,cell_2,hidden_2]
		)

	orig_c1 = P.init_recurrent_1_cell.get_value()
	orig_h1 = T.tanh(P.init_recurrent_1_hidden).eval()
	orig_c2 = P.init_recurrent_2_cell.get_value() 
	orig_h2 = T.tanh(P.init_recurrent_2_hidden).eval()

	for _ in xrange(20):
		result = prime_str
		c_id = -1
		probs,c1,h1,c2,h2 = sample(temp_input,c_id,orig_c1,orig_h1,orig_c2,orig_h2)

		for char in prime_str:
			c = char2id[char]
			probs,c1,h1,c2,h2 = sample(temp_input,c,c1,h1,c2,h2)

		while len(result) < 200:
			probs = probs.astype(np.float64)
			probs = probs / probs.sum()
			c_id = np.argmax(np.random.multinomial(1,pvals=probs))
			char = id2char[c_id]
			if char == "\n": break
			result = result + char
			probs,c1,h1,c2,h2 = sample(temp_input,c_id,c1,h1,c2,h2)
		print result
		

