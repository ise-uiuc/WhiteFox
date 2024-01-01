
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 7, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(7, 101, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(101, 93, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = torch.sigmoid(v1)
        v5 = torch.mul(v3, v4)
        v6 = F.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
