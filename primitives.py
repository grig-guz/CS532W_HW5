import torch
import torch.distributions as dist
from pyrsistent import pmap,pvector, PMap, PList, PVector


class Normal(dist.Normal):

    def __init__(self, alpha, loc, scale):

        if scale > 20.:
            self.optim_scale = scale.clone().detach().requires_grad_()
        else:
            self.optim_scale = torch.log(torch.exp(scale) - 1).clone().detach().requires_grad_()


        super().__init__(loc, torch.nn.functional.softplus(self.optim_scale))

    def Parameters(self):
        """Return a list of parameters for the distribution"""
        return [self.loc, self.optim_scale]

    def make_copy_with_grads(self):
        """
        Return a copy  of the distribution, with parameters that require_grad
        """

        ps = [p.clone().detach().requires_grad_() for p in self.Parameters()]

        return Normal(*ps)

    def log_prob(self, x):

        self.scale = torch.nn.functional.softplus(self.optim_scale)

        return super().log_prob(x)


def push_addr(alpha, value):
    return alpha + value

def plus(addr, arg1, arg2):
    return arg1 + arg2

def minus(addr, arg1, arg2):
    return arg1 - arg2


def product(addr, arg1, arg2):
    return arg1 * arg2

def div(addr, arg1, arg2):
    return arg1 / arg2


def vector(addr, *arglist):
    for arg in arglist:
        if not isinstance(arg, torch.Tensor):
            return pvector(arglist)
    try:
        return torch.stack(arglist, dim=0)
    except Exception:
        return pvector(arglist)

def hash_map(addr, *arglist):
    hmap = pmap({})
    for i in range(0, len(arglist), 2):
        key = arglist[i]
        if isinstance(key, torch.Tensor):
            key = float(key)
        hmap = hmap.update({key: arglist[i+1]})
    return hmap

def get(addr, arg1, arg2):
    if isinstance(arg1, PMap):
        return get_hashmap(addr, arg1, arg2)
    key = get_key(arg2)
    return arg1[int(key)]

def get_hashmap(addr, arg1, arg2):
    key = get_key(arg2)
    return arg1[key]

def put(addr, arg1, arg2, arg3):
    if isinstance(arg1, PMap):
        return put_hashmap(addr, arg1, arg2, arg3)
    key = get_key(arg2)
    if isinstance(arg1, torch.Tensor):
        arg1[int(key)] = arg3
    else:
        arg1 = arg1.set(key, arg3)
    return arg1

def put_hashmap(addr, arg1, arg2, arg3):
    key = get_key(arg2)
    arg1 = arg1.update({key: arg3})
    return arg1

def get_key(arg):
    try:
        key = float(arg)
    except Exception:
        key = arg
    return key

def greater(addr, arg1, arg2):
    return arg1 > arg2

def less(addr, arg1, arg2):
    return arg1 < arg2

def empty(addr, arg1):
    return len(arg1) == 0

def first(addr, arg1):
    return arg1[0]

def last(addr, arg1):
    if isinstance(arg1, torch.Tensor) and arg1.dim() == 0:
        return arg1
    return arg1[-1]

def rest(addr, arg1):
    return arg1[1:]

def conj(addr, arg1, arg2):
    if arg2.dim() == 0:
        arg1 = append(addr, arg1, arg2)
    else:
        for elm in arg2:
            arg1 = append(addr, arg1, elm)
    return arg1

def append(addr, arg1, arg2):
    if isinstance(arg1, PVector):
        arg1 = arg1.append(arg2)
    elif isinstance(arg1, torch.Tensor):
        if arg1.dim() == 0:
            arg1 = arg1.unsqueeze(dim=0)
        if arg2.dim() == 0:
            arg2 = arg2.unsqueeze(dim=0)
        arg1 = torch.cat([arg1, arg2])
    return arg1

def sqrt(addr, arg):
    return torch.sqrt(arg)

def sample(addr, arg1):
    return arg1.sample()

def observe(addr, arg1, arg2):
    return arg2

def beta(addr, arg1, arg2):
    return dist.beta.Beta(arg1, arg2)

def exponential(addr, arg1):
    return dist.exponential.Exponential(arg1)

def uniform(addr, arg1, arg2):
    return dist.uniform.Uniform(arg1, arg2)

def discrete(addr, arg1):
    return dist.categorical.Categorical(arg1)

def flip(addr, arg1):
    return discrete(addr, torch.tensor([1 - arg1, arg1]))

def log(addr, arg1):
    return torch.log(arg1)

env = {
           'normal' : Normal,
           'beta': beta,
           'exponential': exponential,
           'push-address' : push_addr,
           'uniform-continuous': uniform,
           'discrete': discrete,
           'flip': flip,
           '+': plus,
           '-': minus,
           '*': product,
           '/': div,
           'sqrt': sqrt,
           'log': log,
           'vector': vector,
           'get': get,
           'hash-map': hash_map,
           'put': put,
           '>': greater,
           '<': less,
           'empty?': empty,
           'first': first,
           'rest': rest,
           'conj': conj,
           'append': append,
           'sample': sample,
           'observe': observe,
           'peek': last
       }
