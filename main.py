import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math

x_train, y_train = load_data()

print(f"x type: {type(x_train)}")
print(f"first 5 of x: {x_train[:5]}\n")

print(f"Y type: {type(y_train)}")
print(f"first 5 of y: {y_train[:5]}\n")

print(f"x shape: {x_train.shape}")
print(f"y shape: {y_train.shape}\n")

print (f"Number of training examples: {x_train.shape[0]}")
"""
plt.scatter(x_train, y_train, marker='x', color='r')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()
"""
def compute_cost(x, y, w, b): 
    # number of training examples
    m = x.shape[0] 
    
    # You need to return this variable correctly
    total_cost = 0
    
    ### START CODE HERE ###
    sum_cost=0
    for i in range(m):
        fwb = (w*x[i])+b
        cost = (fwb - y[i])**2
        sum_cost=sum_cost+cost
    total_cost=(1/(2*m))*(sum_cost)
    ### END CODE HERE ### 

    return total_cost

# Compute cost with some initial values for paramaters w, b
initial_w = 2
initial_b = 1

cost = compute_cost(x_train, y_train, initial_w, initial_b)
print(type(cost))
print(f'Cost at initial w: {cost:.3f}')

# Public tests
from public_tests import *
#compute_cost_test(compute_cost)

def compute_gradient(x, y, w, b): 
    # Number of training examples
    m = x.shape[0]
    
    # You need to return the following variables correctly
    dj_dw = 0
    dj_db = 0
    
    ### START CODE HERE ###
    dwsum=0
    dbsum=0
    for i in range(m):
        fwb = (w*x[i])+b
        dw = (fwb-y[i])*x[i]
        db = fwb-y[i]
        dwsum+=dw
        dbsum+=db
    dj_dw = (1/m)*(dwsum)
    dj_db = (1/m)*dbsum
    ### END CODE HERE ### 
        
    return dj_dw, dj_db

# Compute and display gradient with w initialized to zeroes
initial_w = 0
initial_b = 0

tmp_dj_dw, tmp_dj_db = compute_gradient(x_train, y_train, initial_w, initial_b)
print('Gradient at initial w, b (zeros):', tmp_dj_dw, tmp_dj_db)

#compute_gradient_test(compute_gradient)

def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    
    # number of training examples
    m = len(x)
    
    # An array to store cost J and w's at each iteration — primarily for graphing later
    J_history = []
    w_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_dw, dj_db = gradient_function(x, y, w, b )  

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            cost =  cost_function(x, y, w, b)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w, b, J_history, w_history #return w and J,w history for graphing

# initialize fitting parameters. Recall that the shape of w is (n,)
initial_w = 0.
initial_b = 0.

# some gradient descent settings
iterations = 1500
alpha = 0.01

w,b,_,_ = gradient_descent(x_train ,y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations)
print("w,b found by gradient descent:", w, b)

m = x_train.shape[0]
predicted = np.zeros(m)

for i in range(m):
    predicted[i] = w * x_train[i] + b

plt.plot(x_train, predicted, c = "b")

plt.scatter(x_train, y_train, marker='x', c='r') 

plt.title("Profits vs. Population per city")

plt.ylabel('Profit in $10,000')
plt.xlabel('Population of City in 10,000s')
plt.show()

predict1 = 3.5 * w + b
print('For population = 35,000, we predict a profit of $%.2f' % (predict1*10000))

predict2 = 7.0 * w + b
print('For population = 70,000, we predict a profit of $%.2f' % (predict2*10000))