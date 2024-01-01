
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
 
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.conv(x)
        v3 = self.conv(x)
        v4 = v2 * v3
        v5 = v1 * 0.044715
        v6 = v5 * 0.7978845608028654
        v7 = v4 + v6
        v8 = v7.tanh()
        v9 = v8 + 1
        v10 = v1 * 0.5
        v11 = v10 * v9
        return v11

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
