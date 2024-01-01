
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        x2 = self.conv(x1)
        x3 = torch.nn.functional.relu6(x2 + 3)
        x4 = torch.div(x3, 6)
        return x4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
