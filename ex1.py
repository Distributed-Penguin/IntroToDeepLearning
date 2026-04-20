import numpy as np
import matplotlib.pyplot as plt

#function model to be learned
def func(x, w, u, b1, b2):
    h = ReLU(x @ u + b1)
    f = h @ w + b2
    return h, f

def main():    
    #training data
    data = np.array([
        [0,0,-1],
        [0,1,1],
        [1,0,1],
        [1,1,-1]
    ])

    #define parameters to be learned
    w = np.random.randn(2)
    u = np.random.randn(2, 2)
    b1 = np.random.randn(2)
    b2 = np.random.randn()
    
    #batch experiment
    batch_epochs = 5000
    batch_lr = 0.001
    batch_learner(data, w.copy(), u.copy(), b1.copy(), b2, batch_epochs, batch_lr)
    
    #mini-batch experiment
    m_batch_epochs = 5000
    m_batch_lr = 0.1
    m_batch_size = 2
    mini_batch_learner(data, w.copy(), u.copy(), b1.copy(), b2, m_batch_epochs, m_batch_lr, m_batch_size)
    
    #SGD experiment
    SGD_epochs = 5000
    SGD_lr = 0.001
    SGD_learner(data, w.copy(), u.copy(), b1.copy(), b2, SGD_epochs, SGD_lr)


def batch_learner(data, w, u, b1, b2, epochs, lr):   
    #parse data
    x = data[:,:-1]
    y = data[:,-1]
    
    #data for iteration plots
    l_ep = np.empty(epochs)
    w_ep = np.empty((epochs, len(w)))
    b2_ep = np.empty(epochs)
    
    #iterate over epochs
    for epoch in range(epochs):
        w, u, b1, b2, l_ep[epoch] = batch_grad_step(x, y, w, u, b1, b2, lr)
        
        #update plot data
        w_ep[epoch] = w
        b2_ep[epoch] = b2
    
    plot_data(l_ep, w_ep, b2_ep, "batch")

def mini_batch_learner(data, w, u, b1, b2, epochs, lr, batch_size):
    #data for iteration plots
    l_ep = np.empty(epochs)
    w_ep = np.empty((epochs, len(w)))
    b2_ep = np.empty(epochs)
    
    #iterate over epochs
    for epoch in range(epochs):
        #sample mini-batch and parse data
        rand_idx = np.random.choice(len(data), size=batch_size, replace=False)
        m_batch = data[rand_idx]
        x = m_batch[:,:-1]
        y = m_batch[:,-1]
        
        w, u, b1, b2, l_ep[epoch] = batch_grad_step(x, y, w, u, b1, b2, lr)
        
        #update plot data
        w_ep[epoch] = w
        b2_ep[epoch] = b2
    
    plot_data(l_ep, w_ep, b2_ep, "mini-batch")

def SGD_learner(data, w, u, b1, b2, epochs, lr):
    data_len = len(data)

    #data for iteration plots
    l_ep = np.empty(epochs * data_len)
    w_ep = np.empty((epochs * data_len, len(w)))
    b2_ep = np.empty(epochs * data_len)
    
    #iterate over epochs
    for epoch in range(epochs):
        for i, sample in enumerate(data):
            x = np.array([sample[:-1]]) #here I wrap each sample in a second list in order to pass off to batch_grad_step. This saves on writing up an extra class with slight efficiency tradeoff
            y = np.array(sample[-1])
            
            w, u, b1, b2, l_ep[(epoch * data_len) + i] = batch_grad_step(x, y, w, u, b1, b2, lr)
            
            #update plot data
            w_ep[(epoch * data_len) + i] = w
            b2_ep[(epoch * data_len) + i] = b2
    
    plot_data(l_ep, w_ep, b2_ep, "SGD")

def batch_grad_step(x, y, w, u, b1, b2, lr):
    data_len = len(x)
        
    #batch-calculate function and appropriate loss
    h, f = func(x, w, u, b1, b2)
    loss = np.sum((y - f) ** 2)
    #batch-calculate dl_df
    dl_df = -2 * (y-f)

    #batch-calculate dl_db2 = dl_df * df_db2 = dl_df * 1
    dl_db2 = dl_df.copy()
    b2_step = lr * np.sum(dl_db2)

    #batch-calculate dl_dw = dl_df * df_dw
    dl_dw = np.array([dl_df[i] * h[i] for i in range(data_len)])
    w_step = lr * np.sum(dl_dw, axis=0)

    #where z = U.T * x + b1, h = ReLU(z), calculate dl_dz = dl_dh * dh_dz  
    dl_dh = np.array([dl_df[i] * w for i in range(data_len)])
    dh_dz = (h > 0).astype(float) #returns an indicator vector where positive values have derivative 1, otherwise - 0
    dl_dz = dl_dh * dh_dz
    
    # batch calculate dl_db1 = dl_dz * dz_db1 = dl_dz
    dl_db1 = dl_dz
    b1_step = lr * np.sum(dl_db1, axis=0)

    #this trick calculates the gradiant matrix, taking into account that xn multiplies row n of u and each row of w multiplies each column of u
    #using matrix multiplication from all samples to all targets (2,4)X(4, 2) yields the sum of entire batch
    u_step = lr * x.T @ dl_dz

    #gradient steps
    w -= w_step
    u -= u_step
    b1 -= b1_step
    b2 -= b2_step
    return w, u, b1, b2, loss

#ReLU implementation
def ReLU(num):
    return np.maximum(num,0)
    
#plot data
def plot_data(l, w, b2, experiment_name):
    plt.figure()
    plt.suptitle(experiment_name)

    plt.subplot(2,2,1)
    plt.plot(range(len(l)),l)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss V Iterations')    

    plt.subplot(2,2,2)
    plt.plot(range(len(w)), w[:,0])
    plt.xlabel('Iteration')
    plt.ylabel('w[0]')
    plt.title('w[0] V Iterations')

    plt.subplot(2,2,3)
    plt.plot(range(len(w)), w[:,1])
    plt.xlabel('Iteration')
    plt.ylabel('w[1]')
    plt.title('w[1] V Iterations')

    plt.subplot(2,2,4)
    plt.plot(range(len(b2)), b2)
    plt.xlabel('Iteration')
    plt.ylabel('b2')
    plt.title('b2 V Iterations')
    
    plt.show()

if __name__ == "__main__":
    main()

