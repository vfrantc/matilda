import math
import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt

__all__ = ['make_filter_bank',
           'show_filter_bank',
           'get_available']


AVAILABLE_TRANSFORMS = ('dct', 'wlsh', 'slant', 'chebychev')

def get_available():
    return AVAILABLE_TRANSFORMS

def _chebychev_filters(n=3, groups=1, expand_dim=1, level=None, DC=True, l1_norm=True):
    # Use one of predefined matrices
    _filters = [(1 / 9) * np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32),
                (1 / 6) * np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32),
                (1 / 6) * np.array([[1, -2, 1], [1, -2, 1], [1, -2, 1]], dtype=np.float32),
                (1 / 6) * np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32),
                (1 / 4) * np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]], dtype=np.float32),
                (1 / 4) * np.array([[-1, 2, -1], [0, 0, 0], [1, -2, 1]], dtype=np.float32),
                (1 / 6) * np.array([[1, 1, 1], [-2, -2, -2], [1, 1, 1]], dtype=np.float32),
                (1 / 4) * np.array([[-1, 0, 1], [2, 0, -2], [-1, 0, 1]], dtype=np.float32),
                (1 / 4) * np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)]

    if level is None:
        filter_bank = np.zeros((n, n, (n**2-int(not DC))), dtype=np.float32)
    else:
        filter_bank = np.zeros((n, n, (level*(level+1)//2-int(not DC))), dtype=np.float32)
    m = 0
    for i in range(n):
        for k in range(n):
            if (not DC and i == 0 and k == 0) or (not level is None and i + k >= level):
                continue

            filter_bank[:, :, m] = _filters[i*n+k]

            if l1_norm:
                filter_bank[:, :, m] /= np.sum(np.abs(filter_bank[:, :, m]))

            m += 1
    filter_bank = np.tile(np.expand_dims(filter_bank, axis=expand_dim), (1,1,1, groups))
    return filter_bank

def _dct_filters(n=3, groups=1, expand_dim=2, level=None, DC=True, l1_norm=True):
    if level is None:
        filter_bank = np.zeros((n, n, (n**2-int(not DC))), dtype=np.float32)
    else:
        filter_bank = np.zeros((n, n, (level*(level+1)//2-int(not DC))), dtype=np.float32)
    m = 0
    for i in range(n):
        for k in range(n):
            if (not DC and i == 0 and k == 0) or (not level is None and i + k >= level):
                continue
            ai = 1.0 if i > 0 else 1.0 / math.sqrt(2.0)
            ak = 1.0 if k > 0 else 1.0 / math.sqrt(2.0)
            for x in range(n):
                for y in range(n):
                    filter_bank[x, y, m] = math.cos((math.pi * (x + .5) * i) / n) * math.cos((math.pi * (y + .5) * k) / n)
            if l1_norm:
                filter_bank[:, :, m] /= np.sum(np.abs(filter_bank[:, :, m]))
            else:
                filter_bank[:, :, m] *= (2.0 / n) * ai * ak
            m += 1
    filter_bank = np.tile(np.expand_dims(filter_bank, axis=expand_dim), (1,1,1,groups))
    return filter_bank

def _wlsh_filters(n=4, groups=1, expand_dim=1, level=None, DC=True, l1_norm=True):
    assert n in (1, 2, 4, 8)

    def hadamard(N):
        if N == 1:
            return np.ones((1, 1), dtype=np.float32)
        else:
            return np.kron(np.array([[1, 1], [1, -1]], dtype=np.float32),
                           hadamard(N // 2))

    H = hadamard(n).tolist()
    H.sort(key=lambda x: sum(map(lambda a: a[0] * a[1] < 0, zip(x[1:], x[:-1]))))
    H = np.array(H, dtype=np.float32)

    if level is None:
        filter_bank = np.zeros((n, n, n**2-int(not DC)), dtype=np.float32)
    else:
        filter_bank = np.zeros((n, n, level*(level+1)//2-int(not DC)), dtype=np.float32)
    m = 0
    for i in range(n):
        for k in range(n):
            if (not DC and i == 0 and k == 0) or (not level is None and i + k >= level):
                continue

            filter_bank[:, :, m] = H[i, :].reshape(n, -1).dot(H[k, :].reshape(-1, n))

            if l1_norm:
                filter_bank[:, :, m] /= np.sum(np.abs(filter_bank[:, :, m]))

            m += 1
    filter_bank = np.tile(np.expand_dims(filter_bank, axis=expand_dim), (1,1,1, groups))
    return filter_bank

def _slant_filters(n=4, groups=1, expand_dim=1, level=None, DC=True, l1_norm=True):
    assert n in (1, 2, 4)
    # Use one of predefined matrices
    _s = {1: np.array([1], dtype=np.float32),
          2: (1 / math.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.float32),
          4: (1 / 2) * np.array([[1, 1, 1, 1],
                                 [3 / math.sqrt(5), 1 / math.sqrt(5), -1 / math.sqrt(5), -3 / math.sqrt(5)],
                                 [1, -1, -1, 1],
                                 [1 / math.sqrt(5), -3 / math.sqrt(5), 3 / math.sqrt(5), -1 / math.sqrt(5)]],
                                dtype=np.float32)}
    S = _s[n]

    if level is None:
        filter_bank = np.zeros((n, n, (n**2-int(not DC))), dtype=np.float32)
    else:
        filter_bank = np.zeros((n, n, (level*(level+1)//2-int(not DC))), dtype=np.float32)
    m = 0
    for i in range(n):
        for k in range(n):
            if (not DC and i == 0 and k == 0) or (not level is None and i + k >= level):
                continue

            filter_bank[:, :, m] = S[i, :].reshape(n, -1).dot(S[k, :].reshape(-1, n))

            if l1_norm:
                filter_bank[:, :, m] /= np.sum(np.abs(filter_bank[:, :, m]))

            m += 1
    filter_bank = np.tile(np.expand_dims(filter_bank, axis=expand_dim), (1,1,1,groups))
    return filter_bank

def show_kernel(kernel, ax):
    height, width = kernel.shape[:2]

    # Limits for the extent
    x_start = 3.0
    x_end = 9.0
    y_start = 6.0
    y_end = 12.0

    extent = [x_start, x_end, y_start, y_end]

    im = ax.imshow(kernel, extent=extent, origin='lower', interpolation='None', cmap='viridis')

    jump_x = (x_end - x_start) / (2.0 * width)
    jump_y = (y_end - y_start) / (2.0 * height)
    x_positions = np.linspace(start=x_start, stop=x_end, num=width, endpoint=False)
    y_positions = np.linspace(start=y_start, stop=y_end, num=height, endpoint=False)

    for y_index, y in enumerate(y_positions):
        for x_index, x in enumerate(x_positions):
            label = '{:6.3f}'.format(kernel[y_index, x_index])
            text_x = x + jump_x
            text_y = y + jump_y
            ax.text(text_x, text_y, label, color='red', ha='center', va='center')
    plt.xticks([])
    plt.yticks([])


def show_filter_bank(fb, figsize=(16, 12)):
    num_filters = fb.shape[-1]
    grd_x = grd_y = int(fb.shape[0])

    fig = plt.figure(figsize=figsize)
    for f_idx in range(num_filters):
        ax = fig.add_subplot(grd_x, grd_y, f_idx + 1)
        kernel = fb[:, :, :, f_idx].squeeze()
        show_kernel(kernel=kernel, ax=ax)
        plt.title('[{}, {}]'.format(f_idx // grd_x, f_idx % grd_x))
    fig.tight_layout()
    plt.show()

def make_filter_bank(ftype='dct', n=4, level=None, DC=True, l1_norm=True):
    assert ftype in AVAILABLE_TRANSFORMS
    fname = ftype.strip().lower()
    filters = None
    if fname == 'dct':
        filters = _dct_filters(n=n, groups=1, expand_dim=2, level=None, DC=DC, l1_norm=l1_norm)
    elif fname == 'wlsh':
        filters = _wlsh_filters(n=n, groups=1, expand_dim=2, level=None, DC=DC, l1_norm=l1_norm)
    elif fname == 'slant':
        filters = _slant_filters(n=n, groups=1, expand_dim=2, level=None, DC=DC, l1_norm=l1_norm)
    elif fname == 'chebychev':
        filters = _chebychev_filters(n=n, groups=1, expand_dim=2, level=None, DC=DC, l1_norm=l1_norm)

    idxs = [n*i+j for i,j in level]
    return filters [..., idxs]

if __name__ == '__main__':
    filters = make_filter_bank(ftype='dct', n=4, level=[(0, 0), (0, 1), (1, 1)])
    show_filter_bank(filters)