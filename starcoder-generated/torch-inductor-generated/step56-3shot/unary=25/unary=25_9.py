
class Model(torch.nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.linear = torch.nn.Linear(num_channels, num_channels)
 
    def forward(self, x1, negative_slope):
        v1 = self.linear(x1)
        v2 = v1 > 0 
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
m = Model(100)

# Inputs to the model
x1 = torch.randn(1, 100)
