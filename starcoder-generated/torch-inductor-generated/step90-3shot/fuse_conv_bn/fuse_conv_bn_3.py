
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module = torch.nn.Linear(64, 100)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, input):
        y0 = self.bn(self.module(input))
        return y0
# Input to the model
input = torch.randn(1, 3, 64, 64)
# Model End

