
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = torch.nn.Sequential(torch.nn.ReflectionPad2d(1), torch.nn.Conv2d(1, 1, (6, 9), stride=(2, 2), bias=False))
    def forward(self, x1):
        y = self.block0(x1)
        return y
# Inputs to the model
x1 = torch.randn(1, 1, 19, 32)
