import numpy as np
import pandas as pd

def gradient_descent(x,y):
    m_curr = b_curr = 0
    iteration = 200000
    n = len(x)
    learning_rate = 0.0001

    for i in range(iteration):
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])
        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2/n)*sum(y-y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        print("m {}, b {}, iteration {}, cost {}".format(m_curr,b_curr,i,cost))


if __name__ == "__main__":
    file = pd.read_csv(r"D:\Hassan\Projects\machine learning\gradient_descent\test_scores.csv")
    x = file.math
    y = file.cs

    gradient_descent(x,y)