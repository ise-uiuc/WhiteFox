
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, stride=2)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 32, 3, stride=2)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.conv2(x))
        return y
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
