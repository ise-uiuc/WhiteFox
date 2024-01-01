
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=0)
        self.conv2 = nn.Conv2d(3, 16, 3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(3, 16, 3, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = self.conv3(x1)
        v4 = (v1 + v2 + v3)/3
        v5 = torch.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
