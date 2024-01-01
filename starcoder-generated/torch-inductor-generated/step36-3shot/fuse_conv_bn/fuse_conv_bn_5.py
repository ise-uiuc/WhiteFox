
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 2)
        self.norm1 = torch.nn.BatchNorm1d(3)
    def forward(self, x1):
        s = self.conv1(x1)
        s = self.conv1(s)
        s = self.norm1(s)
        t = self.norm1(s)
        return s
# Inputs to the model
x1 = torch.randn(1, 3, 5, 5)
