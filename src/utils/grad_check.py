import numpy as np

def grad_check(cost_fn, theta, analytic_grad, epsilon=1e-7):

    grad_central=np.zeros(analytic_grad.shape)
    
    plus_theta=theta.copy()
    minus_theta=theta.copy()

    for i in range(analytic_grad.shape[0]):
        plus_theta[i]+=epsilon
        minus_theta[i]-=epsilon
        
        assert grad_central.shape == analytic_grad.shape

        grad_central[i]=(cost_fn(plus_theta) - cost_fn(minus_theta)) / (2 * epsilon)
        plus_theta=theta.copy()
        minus_theta=theta.copy()

        
    error = np.linalg.norm(grad_central - analytic_grad) / (np.linalg.norm(grad_central) + np.linalg.norm(analytic_grad))
    
    return error

