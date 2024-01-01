
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
      self.avg = torch.nn.AvgPool2d(None, None, stride=1, count_include_pad=True)
    def forward(self, x):
        y = self.avg(x)
        z = self.avg(y)
        return x + y + z
# Inputs to the model
x = torch.randn(1, 1, 4, 4)
