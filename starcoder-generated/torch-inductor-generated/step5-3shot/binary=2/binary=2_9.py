
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 8, 1, stride=1, padding=1)
    def forward(self, x):
        t1 = self.conv1(x)
        t2 = self.conv2(t1)
        t3 = t2 - 1
        return t3
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
