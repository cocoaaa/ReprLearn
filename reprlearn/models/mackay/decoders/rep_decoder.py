from scipy import stats
import numpy as np
def decode(msg, strategy='majority_vote', n_repeat=3) -> np.ndarray:
    "n_repeat: number of repetitions used in Encoder"
    n_starts = len(msg) // n_repeat

    out = []
    for i in range(n_starts):
        i_start = i*n_repeat
        i_end = i_start + n_repeat
        m = stats.mode(msg[i_start:i_end]).mode[0]
        out.append(m)

        # debug
        # print(i_start, list(msg[i_start:i_end]))
        # print(m)
        # breakpoint()
    return np.array(out)

