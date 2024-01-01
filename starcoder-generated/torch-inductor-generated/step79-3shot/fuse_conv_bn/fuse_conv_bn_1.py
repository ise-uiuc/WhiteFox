
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Sequential(torch.nn.Conv2d(4, 8, 1), torch.nn.BatchNorm2d(8))
    def forward(self, x1):
        s1 = (self.layer(x1) + 1).relu()
        s2 = (self.layer(x1) + 1).relu()
        s1 = s1 + s1
        return s1 + s2
# Inputs to the model
x1 = torch.randn(1, 4, 4, 4)
