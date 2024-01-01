
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 8, 1, stride=1, padding=1)
    def forward(self, x1, x2):
        v1 = F.relu6(self.conv1(x1))
        v2 = self.conv2(x2)
        v3 = torch.sum([v1, v2], dim=1)
        v4 = self.conv3(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
x2 = torch.randn(1, 3, 32, 32)
