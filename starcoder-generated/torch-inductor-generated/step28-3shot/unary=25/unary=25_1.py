
class Model(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = v1 * 0.2
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
m = Model(256, 256)

# Inputs to the model
x1 = torch.randn(3, 256)
