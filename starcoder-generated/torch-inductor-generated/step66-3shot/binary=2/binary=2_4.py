
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x0):
        v0 = self.conv(x0)
        return v0
# Inputs to the model
x0 = torch.randn(1, 3, 64, 64)
