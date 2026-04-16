import numpy as np
import matplotlib.pyplot as plt

#ReLU implementation
def ReLU(num):
    return np.maximum(num,0)

#function to be learned
def func(x, w, U, b1, b2):
    h = ReLU(U.T @ x + b1)
    return w @ h + b2

def main(): 

    #batch learning parameters
    epochs = 500
    lr = 0.01

    #labeled data
    x = np.array([
        [0,0],
        [0,1],
        [1,0],
        [1,1]
    ])
    y = np.array([-1, 1, 1, -1])

    #define parameters to be learned
    w = np.random.randn(2)
    u = np.random.randn(2, 2)
    b1 = np.random.randn(2)
    b2 = np.random.randn()

    data_len = len(x)
    
    #data for iteration plots
    l_ep = np.empty(epochs)
    w_ep = np.empty((epochs, len(w)))
    b2_ep = np.empty(epochs)
    
    #iterate over epochs
    for epoch in range(epochs):
        
        #batch-calculate function and appropriate loss
        h = np.array([ReLU(u.T @ x_i + b1) for x_i in x])
        f = np.array([w @ h[i] + b2 for i in range(data_len)])
        
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

        #update plot data
        l_ep[epoch] = np.sum((y - f) ** 2)
        w_ep[epoch] = w
        b2_ep[epoch] = b2

    #plot data
    plt.subplot(2,2,1)
    plt.plot(range(epochs), l_ep)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss V Epochs')    

    plt.subplot(2,2,2)
    plt.plot(range(epochs), w_ep[:,0])
    plt.xlabel('Epoch')
    plt.ylabel('w[0]')
    plt.title('w[0] V Epochs')

    plt.subplot(2,2,3)
    plt.plot(range(epochs), w_ep[:,1])
    plt.xlabel('Epoch')
    plt.ylabel('w[1]')
    plt.title('w[1] V Epochs')

    plt.subplot(2,2,4)
    plt.plot(range(epochs), b2_ep)
    plt.xlabel('Epoch')
    plt.ylabel('b2')
    plt.title('b2 V Epochs')
    
    plt.show()
    

if __name__ == "__main__":
    main()