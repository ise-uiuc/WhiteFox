
## Class Definition
class Model(torch.nn.Module):
    def __init__(self, n_channels, k, s, p):
        super().__init__()
        self.linear = torch.nn.Linear(n_channels, 16*16)
 
    def forward(self, x2):
        v2 = self.linear(x2)
        o2 = v2 - x2
        return o2

# Initializing the model
m = Model(3, 1, 1, 1)

# Inputs to the model
x2 = torch.randn(1, 3, 16, 16)
