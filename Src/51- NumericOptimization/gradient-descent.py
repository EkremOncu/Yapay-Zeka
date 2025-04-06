import numpy as np

def f(x):
    return 3 * x ** 2 - 6 * x + 2

def f_d(x):
    return 6 * x - 6

def gradient_descent(f, f_d, x,  *, niter=None, alpha=1e-5, maxiter=10_000_000):
    if niter != None and niter > maxiter:
        raise ValueError(f'niter must be less than max_iter. Your max_iter is = {maxiter}')
    x_old = x
    count = 0
    for _ in range(maxiter):
        x_new = x_old - alpha * f_d(x_old)       
        if niter != None:
            if count >= niter:
                return x_new, f(x_new)
            count += 1      
        else:
            if abs(f(x_new) - f(x_old)) < 1e-15:
                return x_new, f(x_new)
        x_old = x_new
         
    return None

t = gradient_descent(f, f_d, 10, niter=1000000)
if t != None:
    x, minval = t
    print(x, minval)
else:
    print('no minimum')


def gradient_descent_minimize(df, d2f, low, high, epsilon=1e-1, **kwargs):
    global_minimum = None
    xs = np.linspace(low, high, 100)
    for x0 in xs: 
        t = gradient_descent(df, d2f, x0, **kwargs)
        if t:
            x, minimum = t
            if global_minimum is None:
                global_minimum = minimum
                continue
        if abs(minimum - global_minimum) < epsilon:
            global_minimum = minimum
            
    return x, global_minimum


x, minval = gradient_descent_minimize(f, f_d, -10000, 1000, niter=1000000)
print(x, minval)


    
    
    
    