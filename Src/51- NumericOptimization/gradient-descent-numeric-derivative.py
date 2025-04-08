import numpy as np

def derivative(f, x, h=1e-10):
    return (f(x + h) - f(x - h)) / (2 * h)

def f(x):
    return 3 * x ** 2 - 6 * x + 2

def gradient_descent(f, x,  *, niter=None, alpha=1e-5, maxiter=10_000_000):
    if niter != None and niter > maxiter:
        raise ValueError(f'niter must be less than max_iter. Your max_iter is = {maxiter}')
    x_old = x
    count = 0
    for _ in range(maxiter):
        x_new = x_old - alpha * derivative(f, x_old)       
        if niter != None:
            if count >= niter:
                return x_new, f(x_new)
            count += 1      
        else:
            if abs(f(x_new) - f(x_old)) < 1e-15:
                return x_new, f(x_new)
        x_old = x_new
         
    return None

def gradient_descent_minimize(df, low, high, epsilon=1e-1, **kwargs):
    global_minimum = None
    xs = np.linspace(low, high, 10)
    for x0 in xs: 
        t = gradient_descent(df, x0, **kwargs)
        if t:
            x, minimum = t
            if global_minimum is None:
                global_minimum = minimum
                continue
        if abs(minimum - global_minimum) < epsilon:
            global_minimum = minimum
            
    return x, global_minimum

x, minval = gradient_descent_minimize(f, -10, 100, niter=1000000)
print(x, minval)


    
    
    
    