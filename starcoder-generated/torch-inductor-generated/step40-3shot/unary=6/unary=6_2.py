
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(3, 3, 3)
        self.conv1 = torch.nn.Conv2d(3, 3, 3)
        self.add = torch.nn.Add()
    def forward(self, x1):
        t1 = self.conv0(x1)
        t2 = self.conv1(t1)
        t3 = self.add(t1, t2)
        t4 = t3 / 10
        return t3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
