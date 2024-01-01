
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 60, 14, stride=11, padding=4)
        self.conv2 = torch.nn.Conv2d(60, 80, 4, stride=4, padding=1)
        self.conv3 = torch.nn.Conv2d(80, 3, 2, stride=2, padding=1)
    def forward(self, x1):
        v1 = torch.relu(self.conv1(x1))
        v2 = torch.relu(self.conv2(v1))
        v3 = torch.relu(self.conv3(v2))
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 192, 192)
