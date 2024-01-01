
class Model(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 * torch.clamp(min=0, max=6, input=v1 + 3)
        v3 = v2 / 6
        return v3

# Initializing the model
m = Model(64, 64)

# Inputs to the model
x1 = torch.randn(1, 64)
