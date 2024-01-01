
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, (1, 5), stride=1, padding=(0, 2))
        self.conv2 = torch.nn.Conv2d(3, 8, 4, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 8, (2, 3), stride=1, padding=(1, 1))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = self.conv3(x1)
        v4 = v1 + v2 + v3
        v5 = torch.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
