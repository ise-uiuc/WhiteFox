
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv1 = torch.nn.Conv2d(7, 13, 1, stride=1, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv0(x1)
        v1c = self.conv0(x1)
        v2 = self.conv1(x2)
        v3 = v1 + v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 11, 11)
x2 = torch.randn(1, 7, 11, 11)
