
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Sequential(torch.nn.Conv2d(5, 5, 2), torch.nn.BatchNorm2d(5))
    def forward(self, x1, x2):
        out = [self.layer(x1), self.layer(x2)]
        return out[0], out[1]
# Inputs to the model
x1 = torch.randn(1, 5, 4, 4)
x2 = torch.randn(2, 5, 4, 4)
