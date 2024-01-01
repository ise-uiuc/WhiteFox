
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 2, 1, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(2, 4, 1, stride=1, padding=2)
        self.conv3 = torch.nn.Conv2d(4, 16, 1, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.max_pool2d(v1, 3, stride=2, padding=1)
        v3 = torch.relu(self.conv2(v1))
        v4 = torch.relu(self.conv3(v3))
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 55, 55)
