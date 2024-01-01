
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3, padding=1), torch.nn.BatchNorm2d(3), torch.nn.Conv2d(3, 3, 3, padding=1), torch.nn.BatchNorm2d(3))
    def forward(self, x1):
        s2 = self.layer(x1)
        return s2 + s2
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
