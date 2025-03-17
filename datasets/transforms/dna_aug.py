import torch
import random
import numpy as np
from torchvision import transforms

def DNA_extention(dna, length):

    def extend_str(s, l):
        vocab = "ACGT"
        left_length = random.randint(0, l)
        right_length = l - left_length
        left_extension = ''.join(random.choices(vocab, k=left_length))
        right_extension = ''.join(random.choices(vocab, k=right_length))
        return left_extension + s + right_extension

    if type(dna)==str:
        return extend_str(dna, length)
    elif type(dna)==list:
        res = []
        for s in dna:
            res.append(extend_str(s, length))
        return res
    else:
        raise TypeError("DNA should be str or list of str")


if __name__ == '__main__':
    dna = "123456"
    for i in range(10):
        print(DNA_extention(dna, 2))