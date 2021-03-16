from primitives import env as penv
from daphne import daphne
from tests import is_tol, run_prob_test,load_truth
from pyrsistent import pmap,plist
import operator as op
import torch
import numpy as np
import sys
sys.setrecursionlimit(5000)
print("Recursion limit", sys.getrecursionlimit())
# TODO: Make tensors immutable too
# TODO: Prog 12 is prolly wrong also

def standard_env():
    "An environment with some Scheme standard procedures."
    env = pmap(penv)
    env = env.update({'alpha' : ''})
    return env

class Env(dict):
    "An environment: a dict of {'var': val} pairs, with an outer Env."
    def __init__(self, parms=(), args=(), outer=None):
        self.update(standard_env())
        self.update(zip(tuple(parms), tuple(args)))
        self.outer = outer

    def find(self, var):
        "Find the innermost Env where var appears."
        return self if (var in self) else self.outer.find(var)

class Procedure(object):
    "A user-defined Scheme procedure."
    def __init__(self, parms, body, env):
        self.parms, self.body, self.env = parms, body, env

    def __call__(self, *args):
        return evaluate(self.body, sigma=0, env=Env(self.parms, args, self.env))


def evaluate(exp, sigma=0, env=None):
    #print("Current expression", exp)
    if env is None:
        env = Env()
    if isinstance(exp, str) and exp.startswith('"') and exp.endswith('"'):
        return exp
    elif isinstance(exp, str):
        try:
            return env.find(exp)[exp]
        except Exception:
            new_env = env
            while True:
                print(new_env)
                new_env = new_env.outer
    elif isinstance(exp, (int, float)):
        return torch.tensor(float(exp))

    op, *args = exp
    if op == 'if':               # conditional
        test, conseq, alt = args
        exp = (conseq if evaluate(test, sigma, env) else alt)
        return evaluate(exp, sigma, env)
    elif op == 'push-address':
        (addr, body) = args
        return env['push-address'](addr, body)
    elif op == 'fn':              # definition
        (parms, body) = args
        return Procedure(parms, body, env)
    else:                            # function application
        proc = evaluate(op, sigma, env)
        f_args = [evaluate(arg, sigma, env) for arg in args]
        #print("EVALUATING", proc, f_args)
        """
        if op == 'fac':
            print(proc.body)
            print(proc.parms)
            print(args)
            print("Environments: ")
            env = proc.env
            while True:
                print(env)
                env = env.outer
            raise Exception
        """
        res = proc(*f_args)
        return res


def get_stream(exp):
    while True:
        yield evaluate(exp)


def run_deterministic_tests():

    for i in range(14,14):

        exp = daphne(['desugar-hoppl', '-i', '../CS532-HW5/programs/tests/deterministic/test_{}.daphne'.format(i)])
        print(exp)
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret = evaluate(exp[2])
        print("Ret:", ret)
        try:
            assert(is_tol(ret, truth))
        except:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))

    print('FOPPL Tests passed')

    for i in range(10,13):

        exp = daphne(['desugar-hoppl', '-i', '../CS532-HW5/programs/tests/hoppl-deterministic/test_{}.daphne'.format(i)])
        print(exp)
        truth = load_truth('programs/tests/hoppl-deterministic/test_{}.truth'.format(i))
        ret = evaluate(exp[2])
        print("Ret:", ret)
        try:
            assert(is_tol(ret, truth))
        except:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))

        print('Test passed')
        raise Exception

    print('All deterministic tests passed')



def run_probabilistic_tests():

    num_samples=1e4
    max_p_value = 1e-2

    for i in range(1,7):
        exp = daphne(['desugar-hoppl', '-i', '../CS532-HW5/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))

        stream = get_stream(exp[2])

        p_val = run_prob_test(stream, truth, num_samples)

        print('p value', p_val)
        assert(p_val > max_p_value)

    print('All probabilistic tests passed')


if __name__ == '__main__':

    #run_deterministic_tests()
    #run_probabilistic_tests()

    for i in range(2,5):
        print(i)
        exp = daphne(['desugar-hoppl', '-i', '../CS532-HW5/programs/{}.daphne'.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))
        acc = []
        for _ in range(5000):
            acc.append(evaluate(exp[2]))
        if i == 4:
            with open(str(i) + ".npy", 'wb') as f:
                for j in range(4):
                    part_acc = []
                    for k in range(1000):
                        part_acc.append(acc[k][j].numpy())
        else:
            with open(str(i) + ".npy", 'wb') as f:
                acc = torch.stack(acc)
                np.save(f, acc)
