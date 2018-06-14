import numpy as np


class Link:
    def __init__(self):
        pass


class IdentityLink(Link):
    def __str__(self):
        return "identity"

    @staticmethod
    def f(x):
        return x

    @staticmethod
    def finv(x):
        return x


class LogLink(Link):
    def __str__(self):
        return "log"

    @staticmethod
    def f(x):
        return np.log(x)

    @staticmethod
    def finv(x):
        return np.exp(x)


class LogitLink(Link):
    def __str__(self):
        return "logit"

    @staticmethod
    def f(x):
        return np.log(x/(1-x))

    @staticmethod
    def finv(x):
        return 1/(1+np.exp(-x))


def convert_to_link(val):
    if isinstance(val, Link):
        return val
    elif val == "identity":
        return IdentityLink()
    elif val == "logit":
        return LogitLink()
    elif val == "log":
        return LogLink()
    else:
        assert False, "Passed link object must be a subclass of iml.Link"
