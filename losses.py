import theano.tensor as T
import numpy as np

def get_gaussian_likelihood(comps, X_, mu_, S_, w_, feat_dim):
    _2PI = 2. * np.pi
    comps = T.cast(comps, 'int32')

    mu = mu_[comps, :]
    w = w_[comps]
    S = S_[comps, :, :]

    mu = T.cast(mu, "float32")
    w = T.cast(w, "float32")
    S = T.cast(S, "float32")
    X = T.cast(X_, "float32")
    feat_dim = T.cast(feat_dim, "float32")

    residuals_t = X - mu

    maha_t = T.diagonal(residuals_t.dot(T.nlinalg.matrix_inverse(S)).dot(residuals_t.T))

    likelihood_t = (
            T.nlinalg.det(_2PI * S) ** -0.5
            * (T.exp(-0.5 * maha_t))
    )
    likelihood_t += feat_dim * np.float32(0.)
    return likelihood_t * w
