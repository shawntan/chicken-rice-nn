import sys
import numpy as np
import theano
import theano.tensor as T
import numpy as np
import cPickle as pickle
import vocab
import model
from theano_toolkit.parameters import Parameters

if __name__ == "__main__":
	vocab_file = sys.argv[1]
	model_file = sys.argv[2]
	char2id = vocab.load(vocab_file)
	P = Parameters()
	lang_model = model.build_sampler(P,
			start_id = -1, end_id = char2id["\n"],
			character_count = len(char2id) + 1,
			embedding_size = 20,
			hidden_size = 50
		)
	P.load(model_file)
	temp = T.scalar('temp')
	sequence,updates = lang_model(temp)
	sample = theano.function(inputs=[temp],outputs=sequence,updates=updates)
	idx2char = pickle.load(open(vocab_file))
	for i in xrange(10):	
		seq = sample(np.float32(1))
		res = []
		for c in seq:
			if c == char2id["\n"]:
				break
			else: res.append(idx2char[c])
		print ''.join(res)
