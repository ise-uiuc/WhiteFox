
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(16, 32, 1, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        return torch.sigmoid(v2)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
