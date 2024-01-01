
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
 
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 * 0.044715
        v3 = v1 * 0.5
        v4 = v1 * v3
        v5 = v4 + 1
        v6 = v2 * 0.7978845608028654
        v7 = torch.tanh(v6)
        v8 = v7 + 1
        v9 = v5 * v8
        return v9

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
