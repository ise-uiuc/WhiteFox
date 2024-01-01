
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x):
        # TODO
        return x
# Inputs to the model
x1 = torch.randn(1, 3, 4, 4)
