
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._modules = ['layer']
        s = torch.nn.Sequential(
            torch.nn.Conv2d(4, 8, 2),
            torch.nn.BatchNorm2d(8)
        )
        self.layer = s
    def forward(self, x1):
        y = self.layer(x1)
        return y
# Inputs to the model
x1 = torch.randn(1, 4, 4, 4)
