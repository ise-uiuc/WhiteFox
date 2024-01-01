
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        in_channels = 3
        out_channels = 8
        bias = True
        self.negative_slope = negative_slope
        self.fc = torch.nn.Linear(in_channels, out_channels, bias)
 
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
m = Model(negative_slope=0.5)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
