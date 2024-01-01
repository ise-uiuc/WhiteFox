
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3), torch.nn.BatchNorm2d(3))
    def forward(self, x):
        return self.layer(x)
# Inputs to the model
x = torch.randn(1, 3, 4, 4)
