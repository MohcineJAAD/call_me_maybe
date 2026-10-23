# import numpy as np

# def pick(mask: np.ndarray, logits: np.ndarray) -> int:
#     allowed = np.flatnonzero(mask)
#     if len(allowed) == 0:
#         raise RuntimeError("decoder bug: no token allowed")
#     if len(allowed) == 1:
#         return int(allowed[0])
#     scores = np.where(mask, logits, -np.inf)
#     return int(np.argmax(scores))

# mask = np.array([True, False, True, True])
# logits = np.array([2.0, 9.0, 5.0, 7.0, 99.0])

# size = len(mask)
# scores = np.full(len(logits), -np.inf)
# scores[:size] = np.where(mask, logits[:size], -np.inf)
# print(scores)
# print(int(np.argmax(scores)))
