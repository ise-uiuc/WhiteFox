
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.linear = torch.nn.Linear(32, 16)
 
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.linear(v1.view(100, 32))
        v3 = v2 * 0.5
        v4 = v2 * 0.044715
        v5 = v4 * v2
        v6 = v2 * 0.7978845608028654
        v7 = torch.tanh(v6 + v5)
        v8 = v3 * v7
        v9 = v8 + 1
        return v9

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 32, 16, 16)
