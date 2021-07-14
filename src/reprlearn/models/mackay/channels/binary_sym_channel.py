import numpy as np
# Send through noisy channel (binary, symmetric channel with p(flip) = f)
def transmit(msg, f=0.1) -> np.ndarray:
    out = np.array([x if (np.random.uniform() > f) else (not x) for x in msg])
    return out
