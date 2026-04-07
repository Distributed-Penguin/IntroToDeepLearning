import numpy as np

epochs = 100
lr = 0.01

x_arr = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])
y = np.array([-1, 1, 1, -1])

w = np.randn(2)
u = np.randn(2, 2)
b1 = np.randn(2)
b2 = np.randn()


def ReLU(num):
    return np.maximum(num,0)

def func(x, w, U, b1, b2):
    h = ReLU(U.T @ x + b1)
    return w @ h + b2

def main(): 
    l = np.empty(epochs)
    for epoch in range(4):
        f = np.empty(4)
        dL_df = np.empty(4)
        for i, x in enumerate(x_arr):
            f[i] = func(x, w, u, b1, b2)
            dL_df[i] = -2 * (y-f)
        l[epoch] = sum((y - f) ** 2)

        

if __name__ == "__main__":
    main()