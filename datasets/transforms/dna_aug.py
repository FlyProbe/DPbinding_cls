import torch
import random
import math
import numpy as np
from torchvision import transforms
from collections import Counter


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


def generate_random_dna_sequence(dna_seq):
    """
    根据给定的 DNA 序列生成一段比例相同的随机 DNA 序列。

    :param dna_seq: 原始 DNA 序列，只包含 'A', 'T', 'C', 'G'
    :return: 生成的随机 DNA 序列，同长度且字母比例相同
    """
    # 统计字母比例
    counts = Counter(dna_seq)
    total = len(dna_seq)

    # 计算每个字母的比例
    weights = [counts['A'] / total, counts['T'] / total, counts['C'] / total, counts['G'] / total]
    if max(weights) > 0.9:
        weights = [0.25, 0.25, 0.25, 0.25]

    # 按比例生成随机序列
    random_sequence = random.choices(population=['A', 'T', 'C', 'G'], weights=weights, k=total)

    # 返回为字符串
    return ''.join(random_sequence)



if __name__ == '__main__':
    dna = "123456"
    for i in range(10):
        print(DNA_extension(dna, 2))