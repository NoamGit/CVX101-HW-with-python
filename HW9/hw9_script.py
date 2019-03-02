from functools import partial
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

n = 100
m = 200
np.random.seed(1)
A = np.random.randn(m, n)

STOP_PARAM = 10 ** -5
ALPHA = 0.25
BETA = 0.8


def obj(x, a):
    """
    x.shape = (n,1)
    """
    out = -np.log((1 - a.dot(x))).sum() - np.log((1 - x ** 2)).sum()
    # logger.info("obj:\t")
    return out


def grad(x, a):
    return a.T.dot(1 / (1 - a.dot(x))) + 2 * x / (1 - x ** 2)


def hess(x, a):
    return a.T.dot(np.diag(1/((1-a.dot(x))**2).squeeze())).dot(a) + np.diag((1/((1-x)**2) + 1/((1+x)**2)).squeeze())


def is_in_domain(x, a):
    if not (a.dot(x) <= 1).all() or not (np.abs(x) <= 1).all():
        return False
    return True


def backtracking(obj_func, x, step_size, grad, is_valid_x_func
                 , alpha: float = 0.25, beta: float = 0.8):
    """
    x.shape = step_size.shape = grad.shape = (n,1)
    """
    t = 1
    while t > beta * t:
        if not is_valid_x_func(x + t * step_size):
            t *= beta
            continue
        elif obj_func(x + t * step_size) < obj_func(x) + alpha * t * grad.T.dot(step_size).squeeze():
            break
        t *= beta
    return t


def gradient_decent(obj_func, grad_func, in_domain_func, x0, logger, max_itr=10 ** 4):
    x_itr = x0
    fx = [obj_func(x0)]
    for k in range(max_itr):

        grad_itr = grad_func(x_itr)
        if np.linalg.norm(grad_itr) <= STOP_PARAM:
            logger.info(f"itr:{k}:early stopping!")
            break

        # get step size
        step_size_itr = -grad_itr

        # line search
        t_itr = backtracking(obj_func, x_itr, step_size_itr, grad_itr, in_domain_func, ALPHA, BETA)

        # GD update
        x_itr += t_itr * step_size_itr
        fx += [obj_func(x_itr)]

        logger.info(f"itr:{k}\tmean_step_size:{t_itr * step_size_itr.mean()}\tobj_value:{fx[-1]}")

    return x_itr, fx


def newton_decent(obj_func, grad_func, in_domain_func, hess_func, tol, x0, logger, max_itr=10 ** 4):
    x_itr = x0
    fx = [obj_func(x0)]
    for k in range(max_itr):

        grad_itr = grad_func(x_itr)
        hess_itr = hess_func(x_itr)

        # compute newton step and decrement
        step_size_itr = -np.linalg.inv(hess_itr).dot(grad_itr)
        newton_decrement = grad_itr.T.dot(-step_size_itr)

        if newton_decrement / 2 <= tol:
            logger.info(f"itr:{k}:early stopping!")
            break

        # line search
        t_itr = backtracking(obj_func, x_itr, step_size_itr, grad_itr, in_domain_func, ALPHA, BETA)

        # GD update
        x_itr += t_itr * step_size_itr
        fx += [obj_func(x_itr)]

        logger.info(f"itr:{k}\tmean_step_size:{t_itr * step_size_itr.mean()}\tobj_value:{fx[-1]}")

    return x_itr, fx


def hessian(x, a):
    pass


if "__main__" == __name__:
    mode = "ND"

    x0 = np.zeros((n, 1))
    obj_func = partial(obj, a=A)
    grad_func = partial(grad, a=A)
    in_domain_func = partial(is_in_domain, a=A)

    # gradient descent
    if mode == "GD":
        logger = logging.getLogger("grad_descent")
        # x = np.random.randn(n,1)
        gradient_decent(obj_func, grad_func
                        , in_domain_func, x0, logger=logger, max_itr=10 ** 4)

    # newton descent
    elif mode == "ND":
        hess_func = partial(hess, a=A)
        tol = 10**-6
        logger = logging.getLogger("NewtonDescent")
        newton_decent(obj_func, grad_func,in_domain_func
                        , hess_func, tol, x0, logger=logger, max_itr=10 ** 4)
