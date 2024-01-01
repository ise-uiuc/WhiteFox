
class Model(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.linear = torch.nn.Linear(in_channels, 1)
 
    def forward(self, x, y):
        v1 = self.linear(x)
        v2 = v1 + y
        return v2

# Initializing the model
m = Model(8)

# Inputs to the model
x = torch.randn(2, 8, 64, 64)
y = torch.randn(2, 8, 64, 64)
