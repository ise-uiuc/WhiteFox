
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(240, 640, 1, groups=10, padding=0, stride=1)
        self.conv2 = torch.nn.Conv2d(10, 40, 3, padding=4, stride=2, groups=10)
    def forward(self, x):
        x = self.conv1(x)
        x = torch.tanh(x)
        x = self.conv2(x)
        x = torch.tanh(x)
        return x
# Inputs to the model
x = torch.randn(1, 240, 100, 100)
