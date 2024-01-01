
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.sep1 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=0, groups=8)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v1a = self.sep1(v1) + v1
        return v1a
# Inputs to the model
x1 = torch.randn(1, 3, 28, 28)
