from __future__ import division
import pickle
import util
import load
import sys
from collections import Counter
import matplotlib.pyplot as plt

dotProduct = util.dotProduct
increment = util.increment

def objective_func(x, y, theta, L=1):
    
    f = 0.5*L*dotProduct(theta, theta) + max(0, 1 - y*dotProduct(theta, x))

    return f

def gradient_func(x, y, theta, L):

    grad = Counter(dict([(f, L*v) for f, v in theta.items()]))

    if y*dotProduct(theta, x) < 1:
        increment(grad, -y, x)

    return grad

def generic_gradient_checker(x, y, theta, L, objective_func, gradient_func, epsilon=0.01, tolerance=1e-4):
    true_grad = gradient_func(x, y, theta, L)
    theta_plus = theta.copy()
    theta_minus = theta.copy()

    grad_diff_norm = 0
    tolsq = tolerance*tolerance

    #print theta.keys()

    delta = 1 - y*dotProduct(theta, x)
    print 'delta =', delta

    #Combine words in theta and x, check obj func change for each word
    for dim in set(theta).union(x):
        if x[dim] != 0:
            eps = min(epsilon, abs(delta/(2*y*x[dim])))
        else:
            eps = epsilon
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
   
    print 'grad_diff_norm =', grad_diff_norm
    return True

def pegasos(train, L, T=10, grad_checking=False, test=None):
    w = Counter()
    t = 0
    #data = [(r[:-1], r[-1]) for r in train]

    for epoch in range(T):
        index = 0;
        for x, y in train:
            index += 1
            print '%4d'%index,
            #sys.stdout.write('\r%d'%index)
            #sys.stdout.flush()

            t += 1
            step = 1/(t*L)

            #yx = {f: y*v for f, v in x.items()}

            if grad_checking:
                if not generic_gradient_checker(x, y, w, L, objective_func, gradient_func):
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
        print '%4d'%epoch,
        if test != None:
            print test_error(w, test),
        print ''
            
    return w

def test_error(w, test):
    n = len(test)
    error = 0

    for x, y in test:
        y_hat = dotProduct(w, x)
        
        if y_hat*y < 0:
            error += 1/n
    
    return error
            



if __name__ == "__main__":


    train = pickle.load(open("train.pkl","rb"))
    test = pickle.load(open("test.pkl","rb"))

    print 'training on', len(train),'reviews'
    #print train[0]

    #lambdas = [1, 10, 50, 100, 120, 140, 150, 160]
    lambdas = [120]
    errors = []
    w_list = []
    min = 0

    for i, L in enumerate(lambdas):
        print 'lambda =', L
        #L = 10**(-3 + i)
        w = pegasos(train, L, 10, test=test)
        error = test_error(w, test)
        errors.append(error)
        w_list.append(w)
        if errors[min] < error:
            min = i

    #Check score accuracy for best w
    w = w_list[min]
    hist = []
    for x, y in test:
        hist.append(dotProduct(w, x)*y)

    plt.figure()
    plt.hist(hist, 25)
    plt.show()


