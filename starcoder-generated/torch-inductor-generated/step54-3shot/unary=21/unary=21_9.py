
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 6, stride=3, padding=0)
        self.conv2 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=1)
    def forward(self, x):
        y = self.conv1(x)
        y = torch.tanh(y)
        y = self.conv1(x)
        y = torch.tanh(y)
        y = self.conv2(y)
        return y
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
