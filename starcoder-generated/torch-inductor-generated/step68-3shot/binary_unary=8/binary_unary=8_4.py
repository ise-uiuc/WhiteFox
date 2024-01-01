
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 5, 7, stride=1)
        self.conv2 = torch.nn.Conv2d(5, 10, 5, stride=1)
        self.conv3 = torch.nn.Conv2d(10, 15, 3, stride=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = torch.relu(self.conv3(v2))
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 24, 24)
