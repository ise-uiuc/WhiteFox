
class Model(torch.nn.Module):
    def __init__(self, linear_channels=16):
        super().__init__()
        self.linear_channels = linear_channels
        self.linear = torch.nn.Linear(16, self.linear_channels)
 
    def forward(self, x1, other):
        v1 = self.linear(x1)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
other = torch.randn(1, 16)
