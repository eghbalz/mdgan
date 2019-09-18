import numpy as np
import lasagne

def iterator(inputs, batchsize, eta_d_shared, eta_g_shared, epoch):
    if (epoch >= 500) and (epoch % 500 == 0):
        eta_d_shared.set_value(lasagne.utils.floatX(0.5 * eta_d_shared.get_value()))
        eta_g_shared.set_value(lasagne.utils.floatX(0.5 * eta_g_shared.get_value()))
    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        yield inputs[excerpt]