import numpy as np

def encode(src_msg, strategy='repetition', n_repeat=None) -> np.ndarray:
    if strategy == 'repetition' and n_repeat is None:
        n_repeat = 3

    t = np.array([[x] * n_repeat for x in src_msg])
    return t.flatten()