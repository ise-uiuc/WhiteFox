
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, 2, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(10, 15, 2, stride=1, padding=1)
    def forward(self, x):
        y = self.conv1(x)
        v0 = self.conv2(y)
        return (y, v0)
# Inputs to the model
x = torch.randn(1, 3, 5, 32)
