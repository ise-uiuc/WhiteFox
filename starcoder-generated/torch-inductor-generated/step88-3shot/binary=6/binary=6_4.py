
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v1 = x1.squeeze(-1).squeeze(-1).view(x1.shape[0], -1, x1.shape[3])
        v4 = v3 - 1
        r1 = v3 * torch.tanh(v4)
        v5 = v4 * torch.rsqrt(v3)
        v6 = v5 + 1
        v7 = v1 + v5 + v6
        v8 = v5 - v1
        v9 = torch.sigmoid(v1) + v5
        v10 = v3 + v3 + v5
        v11 = v1 + v8
        v12 = torch.tanh(r1)
        v13 = torch.tanh(v6)
        v14 = v12 + v13
        v15 = (v15.transpose(-1, -2) + v14.transpose(-1, -2)).transpose(-1, -2)
        return v15

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1, 128, 5120)
