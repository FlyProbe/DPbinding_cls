import torch
import random
import math
import numpy as np
from torchvision import transforms

def DNA_extension(dna, length):
    '''

    :param dna: dna sequence, should be str or list of str
    :param length: if int, extend by length; if float, extend by ratio
    :return:
    '''

    def extend_str(s, l):
        vocab = "ACGT"
        left_length = random.randint(0, l)
        right_length = l - left_length
        left_extension = ''.join(random.choices(vocab, k=left_length))
        right_extension = ''.join(random.choices(vocab, k=right_length))
        return left_extension + s + right_extension

    if isinstance(length, int):
        pass
    elif isinstance(length, float):
        length = math.ceil(len(dna) * length)  # 上取整

    # 根据 dna 的类型操作
    if isinstance(dna, str):
        return extend_str(dna, length)
    elif isinstance(dna, list):
        return [extend_str(seq, length) for seq in dna]

    else:
        raise TypeError("DNA should be str or list of str")


if __name__ == '__main__':
    dna = "123456"
    for i in range(10):
        print(DNA_extension(dna, 2))