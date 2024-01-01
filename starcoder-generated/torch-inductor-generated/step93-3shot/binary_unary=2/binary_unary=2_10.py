
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 1, stride=1, padding=0, bias=True)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=True)
        self.conv3 = torch.nn.Conv2d(64, 128, 5, stride=2, padding=2, bias=True)
        self.conv4 = torch.nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=True)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        v5 = v4 - -5
        v6 = F.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
