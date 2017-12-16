from numbers import Number
import math
import torch
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all


class Exponential(Distribution):
    r"""
    Creates a Exponential distribution parameterized by `rate`.

    Example::

        >>> m = Exponential(torch.Tensor([1.0]))
        >>> m.sample()  # Exponential distributed with rate=1
         0.1046
        [torch.FloatTensor of size 1]

    Args:
        rate (float or Tensor or Variable): rate = 1 / scale of the distribution
    """
    has_rsample = True

    def __init__(self, rate):
        self.rate, = broadcast_all(rate)
        batch_shape = torch.Size() if isinstance(rate, Number) else self.rate.size()
        super(Exponential, self).__init__(batch_shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        u = self.rate.new(*shape).uniform_()
        return -u.log() / self.rate.expand(shape)

    def log_prob(self, value):
        self._validate_log_prob_arg(value)
        log = math.log if isinstance(self.rate, Number) else torch.log
        return log(self.rate) - self.rate * value
