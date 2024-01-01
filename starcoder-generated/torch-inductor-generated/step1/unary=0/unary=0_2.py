
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
 
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 * 0.5
        v3 = v1 + 1
        v4 = v2 * v3
        v5 = v1 * v3
        v6 = v1 * v4
        v7 = v1 * v5
        v8 = v1 + v2
        v9 = v8 * 0.044715
        v10 = v7 * 0.7978845608028654
        v11 = v9 + v10
        v12 = torch.tanh(v11)
        v13 = v5 + 1
        v14 = v12 * v13
        v15 = v3 + 1
        v16 = v13 * v15
        v17 = v14 + v16
        return v17

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
