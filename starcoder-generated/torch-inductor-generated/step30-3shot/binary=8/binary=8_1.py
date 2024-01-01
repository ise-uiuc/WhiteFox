
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 8, 1, stride=1, padding=1)
 
    def forward(self, x1, x2 = None):
        v1 = self.conv(x1)
        if x2 is not None:
            v1 = v1 + x2
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2, 64, 64)
x2 = torch.randn(1, 8, 64, 64)
