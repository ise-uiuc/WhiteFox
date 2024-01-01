
class Model(nn.Module):
    def __init__(self):
      super().__init__()
      self.Conv_2 = nn.Conv2d(3, 1, 1, stride=2,groups=1, bias=False)
    def forward(self, x_t):
        t1 = self.Conv_2(x_t)
        t2 = torch.sigmoid(t1)
        return t2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
