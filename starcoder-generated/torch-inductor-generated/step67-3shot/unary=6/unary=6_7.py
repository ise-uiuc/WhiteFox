
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, (1, 5), stride=1, padding=1, groups=1)
        self.conv2 = torch.nn.Conv2d(8, 16, (1, 1), stride=1, padding=0, groups=1)
    def forward(self, x1):
        t1 = self.conv1(x1)
        t2 = self.conv2(t1)
        return t2
# Inputs to the model
x1 = torch.randn(4, 3, 64, 64)
