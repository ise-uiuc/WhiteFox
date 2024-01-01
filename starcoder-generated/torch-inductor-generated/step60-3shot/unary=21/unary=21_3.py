
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 6, stride=3, padding=0)
        self.conv3 = torch.nn.Conv2d(32, 32, 3, stride=3, padding=0)
    def forward(self, x):
        y = self.conv1(x)
        y = torch.relu(y)
        y = self.conv1(x)
        y = torch.tanh(y)
        y = self.conv3(y)
        return y
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
