
def convolution(t):
    v1 = torch.nn.functional.conv2d(t, weight=torch.ones_like(t),
                                        bias=torch.ones_like(t))
    return v1
 
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = convolution(x1)
        v3 = v1 + v2
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
