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

if __name__ == '__main__':
	counter = CountVectorizer(
			preprocessor=preprocessor,
			tokenizer=tokenise,
			min_df=2
		)
	data_file = open(sys.argv[1],'r')
	counter.fit(data_file)

	print len(counter.vocabulary_)
	pickle.dump(counter.vocabulary_,open(sys.argv[2],'wb'),2)

