import numpy as np
import itertools

def load_db():
    max_x= max_y = 5
    min_x = min_y = -4
    step = 2
    n = 5
    num_data = 10000
    standard_deviation = 0.05
    variance = standard_deviation ** 2
    mus = np.array([np.array([i, j]) for i, j in itertools.product(np.arange(min_x, max_x, step),
                                                                   np.arange(min_y, max_y, step))], dtype=np.float32)
    # init
    nx, ny = (n, n)
    x = np.linspace(min_x, max_x, nx)
    y = np.linspace(min_y, max_y, ny)
    xv, yv = np.meshgrid(x, y)
    N  = xv.size
    num_data_per_mixture = num_data // N

    Xs = []
    for i in range(N):
        d = np.random.multivariate_normal(mus[i], np.eye(2)*variance,num_data_per_mixture)
        Xs.append(d)
    X = np.concatenate(Xs).astype(np.float32)
    return X
