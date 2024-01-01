
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1, __other__=None):
        v1 = self.conv(x1)
        v2 = v1 + __other__
        return v2

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(1, 3, 64, 64)

# A different valid tensor for the model to use
s = torch.randn(1, 8, 64, 64)
