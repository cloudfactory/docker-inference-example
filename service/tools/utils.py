import numpy as np


def rle_encoding(x, transpose=False):
    """
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    """
    if transpose:
        dots = np.where(x.T.flatten() == 1)[0]  # Order down-then-right
    else:
        dots = np.where(x.flatten() == 1)[0]  # Order right-then-down
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def mask2bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax + 1, cmin, cmax + 1
