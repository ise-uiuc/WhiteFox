
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 7, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(32, 64, 5, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, stride=1, padding=2)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v1 = v1.clone()
        v1 = self.conv1(v1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = v3 + v1
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 20, 20)
x2 = torch.randn(1, 32, 20, 20)
