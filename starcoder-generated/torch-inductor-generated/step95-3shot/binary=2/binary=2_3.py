
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - False
        return v2
# Inputs to the model
x1 = torch.randn(1, 8, 28, 28)
