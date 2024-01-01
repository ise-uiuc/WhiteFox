
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3), torch.nn.BatchNorm2d(3))
    def forward(self, x2):
        y1 = self.layer(x2)
        return y1
# Inputs to the model
x2 = torch.randn(1, 3, 4, 4)
