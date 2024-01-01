
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.batch = torch.nn.BatchNorm2d(64)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.batch(v1)
        v3 = nn.Sigmoid()(v2)
        v4 = self.conv2(v3)
        return nn.ReLU()(v4)
# Input to the model
x1 = torch.randn(1, 64, 32, 32)
