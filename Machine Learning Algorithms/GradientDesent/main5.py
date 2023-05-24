import numpy as np

x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])

def gradient_descent(x,y):
    x = np.array([1,2,3,4,5])
    y = np.array([5,7,9,11,13])

    m_curr = b_curr = 0
    iteration = 1000
    n = len(x)
    learning_rate = 0.07

    for i in range(iteration):
        y_predicted = m_curr * x + b_curr
        md = -(2/n) * sum(x*(y-y_predicted))
        bd = -(2/n) * sum(y-y_predicted)
        m_curr = m_curr - learning_rate *md
        b_curr = b_curr - learning_rate *bd
        cost = (1/n) * sum ([val**2 for val in (y-y_predicted)])
        print("m{},b{},iteration{},cost{}".format(m_curr,b_curr,cost,i))


gradient_descent(x,y)