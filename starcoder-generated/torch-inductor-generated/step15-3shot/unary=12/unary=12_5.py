
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 5, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 64, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = torch.sigmoid(self.conv1(x1))
        v2 = torch.mul(self.conv2(v1), v1)
        v3 = torch.mul(self.conv3(v2), v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
