
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)
    def forward(self, x):
        x1 = self.conv(x)
        y2 = bn_layer(x1)
        return y2

def bn_layer(x):
    return False
# Inputs to the model
x = torch.randn(1, 3, 4, 4)
