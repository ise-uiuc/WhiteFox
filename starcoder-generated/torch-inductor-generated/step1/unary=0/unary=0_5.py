
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3, stride=1, padding=1)
 
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 * 0.5
        v3 = self.conv(x)
        v4 = self.conv(x) * v3
        v5 = self.conv(x) * v3 * 0.044715
        v6 = v5 + v4
        v7 = v2 * v6
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        return v9

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
