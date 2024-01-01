
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3 = torch.nn.Conv2d(3, 32, 3, stride=2, padding=0)
        self.conv4 = torch.nn.Conv2d(3, 32, 3, stride=2, padding=0)
        self.conv5 = torch.nn.Conv2d(3, 32, 3, stride=2, padding=0)
        self.conv6 = torch.nn.Conv2d(3, 32, 3, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv3(x1)
        v2 = self.conv4(x1)
        v3 = self.conv5(x1)
        v4 = self.conv6(x1)
        v5 = v1 + v2 + v3 + v4
        v6 = torch.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
