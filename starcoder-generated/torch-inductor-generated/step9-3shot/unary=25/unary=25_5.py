
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        output_channels = 8
        self.lin = torch.nn.Linear(1 + negative_slope, output_channels, bias=True)
        self.negative_slope = negative_slope
 
    def forward(self, x1):
        v1 = self.lin(x1)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
m = Model(negative_slope=(1.0 / 2.0))

# Input to the model
x = torch.randn(1, 1 + (-0.5), 64, 64)
