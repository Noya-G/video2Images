import cv2
import numpy as np


def ahash(image, hash_size=8):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size, hash_size))
    avg = resized.mean()
    ahash_value = sum([2 ** i for i, v in enumerate(resized.flatten()) if v > avg])
    return ahash_value


def phash(image, hash_size=32, highfreq_factor=4):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size, hash_size))
    dct = cv2.dct(np.float32(resized))
    dct_low_freq = dct[:hash_size // highfreq_factor, :hash_size // highfreq_factor]
    median_val = np.median(dct_low_freq)
    phash_value = sum([2 ** i for i, v in enumerate(dct_low_freq.flatten()) if v > median_val])
    return phash_value


def dhash(image, hash_size=8):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


def hamming_distance(hash1, hash2):
    return bin(hash1 ^ hash2).count('1')


