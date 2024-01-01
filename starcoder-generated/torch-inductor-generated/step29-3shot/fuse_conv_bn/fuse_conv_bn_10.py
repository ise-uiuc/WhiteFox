
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [torch.nn.Conv2d(5, 5, 2),
                       torch.nn.Identity(),
                       torch.nn.BatchNorm2d(5)]
        self.seq = torch.nn.Sequential(*self.layers)
    def forward(self, x):
        return self.seq(x)
# Inputs to the model
x = torch.randn(2, 5, 4, 4)
