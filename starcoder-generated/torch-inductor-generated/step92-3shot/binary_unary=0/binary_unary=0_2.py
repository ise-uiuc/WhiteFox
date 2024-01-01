
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, group=10)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, group=10)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, group=10)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v18 = v1 + v2
        v19 = torch.relu(v18)
        v3 = self.conv3(v19)
        v22 = v2 + v3
        v23 = torch.relu(v22)
        return v23
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
