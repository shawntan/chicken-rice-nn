import re
import sys
import cPickle as pickle

import numpy as np
import random


def stream(filename, char2id):
    for line in open(filename):
        charseq = [char2id[c] for c in line]
        charseq.insert(0, -1)
        yield charseq


def randomise(stream, buffer_size=100):
    buf = buffer_size * [None]
    ptr = 0
    for item in stream:
        buf[ptr] = item
        ptr += 1
        if ptr == buffer_size:
            random.shuffle(buf)
            for x in buf:
                yield x
            ptr = 0
    buf = buf[:ptr]
    random.shuffle(buf)
    for x in buf:
        yield x


def sortify(stream, key, buffer_size=200):
    buf = buffer_size * [None]
    ptr = 0
    for item in stream:
        buf[ptr] = item
        ptr += 1
        if ptr == buffer_size:
            buf.sort(key=key)
            for x in buf:
                yield x
            ptr = 0
    buf = buf[:ptr]
    buf.sort(key=key)
    for x in buf:
        yield x


def batch(stream, batch_size=10):
    batch = []
    for item in stream:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch
