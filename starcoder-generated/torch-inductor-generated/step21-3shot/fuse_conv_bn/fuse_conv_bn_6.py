
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 1, bias=False), torch.nn.BatchNorm2d(1))

    def forward(self, x):
        x = self.layer(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 1, 1)
