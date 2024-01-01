
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(6, 6, 2, stride=1, padding=2)
        self.conv1 = torch.nn.Conv2d(6, 6, 2, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(6, 6, 2, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.conv0(x1)
        v2 = self.conv1(v1)
        v3 = self.conv2(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 6, 112, 112)
