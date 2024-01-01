
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(8, 16, 5, stride=2, padding=2)
        self.conv3 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v1 = v3 - v1
        v2 = F.relu(v1)
        return v2
# Inputs to the model
x3 = torch.randn(2, 3, 16, 32)
