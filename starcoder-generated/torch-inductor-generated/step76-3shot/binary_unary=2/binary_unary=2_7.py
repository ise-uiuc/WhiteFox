
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(F.relu(x1 + 3.14))
        v2 = F.avg_pool2d(v1, 2, stride=2)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
