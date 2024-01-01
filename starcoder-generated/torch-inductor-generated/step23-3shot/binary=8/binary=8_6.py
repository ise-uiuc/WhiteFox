
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 128, 1, stride=1, padding=0)
 
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + __other__
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)

# Specify "other"
__other__ = torch.randn(1, 128, 224, 224)
