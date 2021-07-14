import numpy as np

class Encoder():

    def encode(self, msg) -> np.ndarray:
        raise NotImplementedError


class RepEndoer(Encoder):
    """Repetition Encoder"""
    def __init__(self, n_repeat: int):
        self.n_repeat = n_repeat

    def encode(self, msg) -> np.ndarray:
        t = np.array([[x] * self.n_repeat for x in msg])
        return t.flatten()


class LinearBlockEncoder(Encoder):
    pass


class HammingBlockEncoder(Encoder):
    """(7,4) Hamming block encoder
    Encode every K bits into a N-bit-long codeword.
    For (7,4) block code (ie. a type of encoding strategy to encode every K=4 bits in the source msg
    to N=7 sequence by: copy the K source bits as it to the first K bits of the transmitted code, and
    then set every parity bit (one of N-K) to make the parity of sum(the 3 source code bits) = 0
    See. Figure 1.13 on p.9.
    """
    def __init__(self):
        self.N = 7
        self.K = 4
        # "(Code) Generator matrix". this is a bit conflicting because this matrix entails the "encoding" stragegy.
        g_parity = np.array([[1, 1, 1, 0], [0, 1, 1, 1], [1, 0, 1, 1]])
        self.Gt = np.vstack( (np.eye(self.K), g_parity))

    def encode(self,msg) -> np.ndarray:
        out = []
        n_starts = len(msg)//self.K

        for i in range(n_starts):
            i_start = i * self.K
            i_end = i_start + self.K
            out.append(self.Gt.dot(msg[i_start:i_end]))
        return np.array(out).flatten()
