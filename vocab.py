import cPickle as pickle
import sys

def load(filename):
	data = pickle.load(open(filename))
	return { v:i for i,v in enumerate(data) }


if __name__ == "__main__":
	train_file = sys.argv[1]
	vocab_file = sys.argv[2]
	charset = set()
	for line in open(train_file):
		charset.update(line)
	pickle.dump(sorted(charset),open(vocab_file,'wb'),2)
		 
