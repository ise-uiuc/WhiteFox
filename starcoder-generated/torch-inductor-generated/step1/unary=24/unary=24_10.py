
class Model(torch.nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.negative_slope = negative_slope
 
    def forward(self, x):
        v1 = self.conv(x)
        return torch.where(v1 > 0, v1, v1 * self.negative_slope)      

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
