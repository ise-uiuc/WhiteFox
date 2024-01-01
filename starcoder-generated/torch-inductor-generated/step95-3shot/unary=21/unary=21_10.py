
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(3, 24, 1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = x+v1
        return v2
# Inputs to the model
x = torch.randn(1, 3, 208, 304)
