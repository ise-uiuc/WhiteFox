
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = torch.maxpool2d(v2)
        v4 = v3 + torch.abs(torch.randn(1, 1, 1, 1) - 1)
        return v4
