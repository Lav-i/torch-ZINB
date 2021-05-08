import numpy as np
import torch


def _nan2zero(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x), x)


def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)


# def _nelem(x):
#     nelem = tf.reduce_sum(tf.cast(~tf.is_nan(x), tf.float32))
#     return tf.cast(tf.where(tf.equal(nelem, 0.), 1., nelem), x.dtype)
#
#
# def _reduce_mean(x):
#     nelem = _nelem(x)
#     x = _nan2zero(x)
#     return tf.divide(tf.reduce_sum(x), nelem)
#
#
# def mse_loss(y_true, y_pred):
#     ret = tf.square(y_pred - y_true)
#     return _reduce_mean(ret)
#
#
# def mse_loss_v2(y_true, y_pred):
#     y_true = tf.log(y_true + 1)
#     y_pred = tf.log(y_pred + 1)
#     ret = tf.square(y_pred - y_true)
#     return _reduce_mean(ret)


class NB(torch.nn.Module):
    # {\it{{\rm{NB}}}}\left( {{\it{x}};{\it{\mu }},{\it{\theta }}} \right) = \frac{{{\it{\Gamma }}\left( {{\it{x}} + {\it{\theta }}} \right)}}{{{\it{\Gamma }}\left( {\it{\theta }} \right)}}\left( {\frac{{\it{\theta }}}{{{\it{\theta }} + {\it{\mu }}}}} \right)^{\it{\theta }}\left( {\frac{{\it{\mu }}}{{{\it{\theta }} + {\it{\mu }}}}} \right)^{\it{x}}

    def __init__(self, scale_factor=1.0):
        super(NB, self).__init__()
        self.eps = 1e-10
        self.scale_factor = scale_factor

    def forward(self, theta, y_true, y_pred, mean=True):
        y_pred = y_pred * self.scale_factor

        # theta = torch.minimum(theta, torch.tensor(1e6))

        t1 = torch.lgamma(theta + self.eps) + torch.lgamma(y_true + 1.0) - torch.lgamma(y_true + theta + self.eps)
        t2 = (theta + y_true) * torch.log(1.0 + (y_pred / (theta + self.eps))) + (
                y_true * (torch.log(theta + self.eps) - torch.log(y_pred + self.eps)))

        final = t1 + t2

        final = _nan2inf(final)

        if mean:
            final = torch.mean(final)

        return final


class ZINB(NB):
    # {{\text{ZINB}}}\left( {{\it{x}};{\it{\pi }},{\it{\mu }},{\it{\theta }}} \right) = {\it{\pi \delta }}_0\left( {\it{x}} \right) + \left( {1 - {\it{\pi }}} \right){\text{NB}}\left( {{\it{x}};{\it{\mu }},{\it{\theta }}} \right)

    def __init__(self, scale_factor=1.0, ridge_lambda=0.0):
        super().__init__(scale_factor=scale_factor)
        self.ridge_lambda = ridge_lambda
        self.scale_factor = scale_factor

    def forward(self, pi, theta, y_true, y_pred, mean=True):
        nb_case = super().forward(theta, y_true, y_pred, mean=False) - torch.log(1.0 - pi + self.eps)

        y_pred = y_pred * self.scale_factor
        # theta = torch.minimum(theta, torch.tensor(1e6))

        zero_nb = torch.pow(theta / (theta + y_pred + self.eps), theta)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + self.eps)
        result = torch.where(torch.lt(y_true, 1e-8), zero_case, nb_case)

        if self.ridge_lambda > 0:
            ridge = self.ridge_lambda * torch.square(pi)
            result += ridge

        if mean:
            result = torch.mean(result)

        result = _nan2inf(result)

        return result
