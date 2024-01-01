
class Model(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels)
 
    def forward(self, x):
        v = self.linear(x)
        v = v + 3
        v = torch.clamp_min(v, 0)
        v = torch.clamp_max(v, 6)
        v = v / 6
        return v

# Initializing the model
m = Model(2, 2)

# Inputs to the model
x = torch.randn(1, 2)
