
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.split = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.linear = torch.nn.Linear(8, 3)
 
    def forward(self, x):
        v1 = self.split(x)
        v2 = [v1, v1, v1]
        v3 = torch.cat(v2, 1)
        v4 = v3.shape
        v5 = v4[2]
        v6 = v4[3]
        v7 = v3[:, :, :, 0:1]
        v8 = v3[:, :, :, 1:2]
        v9 = v3[:, :, :, 2:3] 
        v10 = v7 + 0.9
        v11 = v8 + 0.8
        v12 = v9 + 0.7
        v13 = v10 + 0.6
        v14 = v11 - 0.5
        v15 = v12 * 0.8
        v16 = v13 * 0.7
        v17 = v15 * 0.6
        v18 = v16 + 0.5
        return v14 + v17 + v18

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
