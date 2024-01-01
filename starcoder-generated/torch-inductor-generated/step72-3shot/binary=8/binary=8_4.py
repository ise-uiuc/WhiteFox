
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 64, 2, stride=2, padding=1)
    def forward(self, x):
        i1 = self.conv1(x)
        i1 = self.conv2(x)
        return i1
# Inputs to the model
x = torch.randn(1, 3, 28, 28)
