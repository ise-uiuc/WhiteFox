
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3, stride=1, padding=1)
 
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 + v1
        v3 = v2 * v2
        v4 = v2 * v2
        v5 = v4 * v4
        v6 = torch.tanh(v5)
        v7 = v3 * 0.7978845608028654
        v8 = v6 + 1
        v9 = v1 * v8
        v10 = v1 * 0.044715
        v11 = v9 * v10
        v12 = v7 - v11
        return v12

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 512, 512)
