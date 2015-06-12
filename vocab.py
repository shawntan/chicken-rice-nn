import argparse
import cPickle as pickle
import sys

def load(filename):
    data = pickle.load(open(filename))
    return {v: i for i, v in enumerate(data)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract unique characters in dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'data_file',
        type=argparse.FileType('r'),
        help="Text file for training."
    )
    parser.add_argument(
        'vocab_file',
        type=argparse.FileType('wb'),
        help="Output file to be used by train.py."
    )
    args = parser.parse_args()


    charset = set()
    for line in args.data_file:
        charset.update(line)
    pickle.dump(sorted(charset), args.vocab_file, 2)
    args.vocab_file.close()
