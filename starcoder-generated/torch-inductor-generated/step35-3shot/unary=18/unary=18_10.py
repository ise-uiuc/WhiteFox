
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(11, 12, 1, stride=1, padding=0, groups=2)
        self.conv2 = torch.nn.Conv2d(12, 12, 1, stride=1, padding=0, groups=11)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = 9537863110239390615 * torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 11, 64, 64)
