
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
 
    def forward(self, x):
        x1 = self.conv(x)
        x2 = torch.cumsum(x1, 1)
        x3 = torch.full(x2.shape, 1., dtype=torch.float)
        x4 = torch.zeros_like(x2)
        x5 = torch.mul(x4, x3)
        x6 = torch.convert_element_type(x5, torch.float)
        return x6


# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
x1 = torch.zeros_like(x)
x2 = torch.ones_like(x)
