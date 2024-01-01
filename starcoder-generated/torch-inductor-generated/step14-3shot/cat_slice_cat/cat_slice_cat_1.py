
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        a1 = torch.cat([x1, x1], 0)
        a2 = a1[:, 0: 9223372036854775807]
        a3 = a2[:, 0:447]
        a4 = torch.cat([a1, a3], 1)
        return a4

# Initializing the model
m = Model()

import numpy as np
# Inputs to the model
x1=np.array([1,2,3,4,5]).astype(np.float32)
x2=np.array([1,2,3,4,5]).astype(np.float32)
x3=np.array([1,2,3,4,5]).astype(np.float32)
x4 = torch.tensor(np.array(x1).reshape(1, x1.shape[0]))
x5 = torch.tensor(np.array(x2).reshape(1, x2.shape[0]))
x6 = torch.tensor(np.array(x3).reshape(1, x3.shape[0]))

x =torch.cat([x4,[x5],x6],0)
x_cut=x[:,0:447]
y = torch.cat([x, x_cut], 1)
y = np.array(y)

print(y.shape)
print(y)
