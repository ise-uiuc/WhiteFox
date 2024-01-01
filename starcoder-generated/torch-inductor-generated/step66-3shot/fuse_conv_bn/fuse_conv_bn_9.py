
class Model(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(2, 2, 1)
        self.conv2 = nn.Conv2d(2, 2, 3)
        self.bn = nn.BatchNorm2d(2)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn(x)
        return x
# Inputs to the model
x = torch.randn(1, 2, 4, 4)
