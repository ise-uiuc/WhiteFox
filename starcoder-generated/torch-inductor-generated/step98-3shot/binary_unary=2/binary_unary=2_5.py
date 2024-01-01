
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=2)
        self.conv2 = torch.nn.Conv2d(8, 8, 3, stride=1)
        self.conv3 = torch.nn.Conv2d(8, 8, 3, stride=1)
    def forward(self, x1):
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        x6 = self.conv3(x3)
        x4 = x6 - 1
        x5 = F.relu(x4)
        return x5
# Inputs to the model
x1 = torch.randn(1, 3, 6, 6)
