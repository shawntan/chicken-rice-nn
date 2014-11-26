# coding=utf-8
import sys,re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import wordpunct_tokenize
import cPickle as pickle

def tokenise(line):
	tokens = line.split(' ')
	tokens.insert(0,'<START>')
	tokens.append('<END>')
	return tokens

def preprocessor(line):
	line = line.strip()
	line = re.sub('[0-9]',"#",line)
	line = line.lower()
	return line

def read_file(filename):
	data_file = open(filename,'r')
	for line in data_file:
		yield tokenise(preprocessor(line))

def count(sentences):
	count = {}
	for s in sentences:
		for w in s:
			count[w] = count.get(w,0) + 1
	return count


		

if __name__ == '__main__':
	data_file  = sys.argv[1]
	top_count  = int(sys.argv[2])
	vocab_file = sys.argv[3]
	count_dict = count(read_file(data_file))
	count_pairs = count_dict.items()
	count_pairs.sort(key=lambda x:-x[1])
	min_df = count_pairs[top_count][1]
	
	rank = top_count
	while count_pairs[rank][1] >= min_df: rank += 1
	vocab = [ word for word,_ in count_pairs[:rank] ]
	pickle.dump(vocab,open(vocab_file,'wb'),2)

	print "Vocabulary file %s generated with %d words."%(vocab_file,len(vocab))
