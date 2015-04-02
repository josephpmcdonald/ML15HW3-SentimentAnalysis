from __future__ import division
import pickle
import util
import load
from collections import Counter

dotProduct = util.dotProduct
increment = util.increment

def objective_func(x, y, theta, L):
    
    f = 0.5*L*dotProduct(theta, theta) + max(0, 1 - y*dotProduct(theta, x))

    return f

def gradient_func(x, y, theta, L):

    grad = dict([(f, L*v) for f, v in theta.items()])

    if y*dotProduct(theta, x) < 1:
        increment(grad, -y, x)

    return grad

def generic_gradient_checker(x, y, theta, L, objective_func, gradient_func, epsilon=0.01, tolerance=1e-4):
    
    true_grad = gradient_func(x, y, theta, L)
    theta_plus = theta.copy()
    theta_minus = theta.copy()

    grad_diff_norm = 0
    tolsq = tolerance*tolerance

    for dim in theta:
        theta_plus[dim] += eps
        theta_minus[dim] -= eps
        plus_eps = objective_func(x, y, theta_plus)
        minus_eps = objective_func(x, y, theta_minus)
        diff = (true_grad[dim] - (plus_eps - minus_eps)/(2*eps))
        grad_diff_norm += diff*diff
        if grad_diff_norm > tolsq:
            return False

        theta_plus[dim] = theta[dim]
        theta_minus[dim] = theta[dim]
    
    return True

def pegasos(L, train, T=100):
    w = {}
    t = 0
    #data = [(r[:-1], r[-1]) for r in train]

    for epoch in range(T):
        print epoch
        index = 0;
        for x, y in train:
            print str(index),
            index += 1

            t += 1
            step = 1/(t*L)

            #yx = {f: y*v for f, v in x.items()}

            if generic_gradient_checker(x, y, w, L, objective_func, gradient_func):
                print 'Gradient check failed'
                return

            if y*dotProduct(w, x) < 1:
                for f, v in w.items():
                    v = (1 - 1/t)*v
                increment(w, step*y, x)
            else:
                for f, v in w.items():
                    v = (1 - 1/t)*v
            
            print '\r',
        w = dict([(f, v) for f, v in w.items() if v != 0])

    return w


if __name__ == "__main__":

    train = pickle.load(open("train.pkl","rb"))
    test = pickle.load(open("test.pkl","rb"))

    print 'training'
    #print train[0]
    w = pegasos(1, train, 1)



