
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(64, 16, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.softmax(v3)
        v5 = self.conv3(v4)
        v6 = torch.softmax(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 1, 1024, 1024)
