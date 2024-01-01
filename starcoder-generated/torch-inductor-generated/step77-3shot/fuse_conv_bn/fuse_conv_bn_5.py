
class Model3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)
        self.conv2 = torch.nn.Conv2d(20, 10, 5, 1)
    def forward(self, x):
        s = self.conv1(x)
        t1 = self.conv2(s)
        return t1
# Inputs to the model
x = torch.randn(1, 1, 32, 32)
