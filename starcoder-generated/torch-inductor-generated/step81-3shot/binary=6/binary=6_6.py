
n_channels = 8
class Model(torch.nn.Module):
    def __init__(self, n_channels=8):
        super().__init__()
        self.linear = torch.nn.Linear(16, n_channels)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - 0.5
        return v2

# Initializing the model
m = Model(n_channels)

# Inputs to the model
x1 = torch.randn(1, 16)
