
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.a1 = nn.Conv2d(1, 3, 3, bias=False)
        self.a2 = torch.nn.BatchNorm2d(3, eps=0.007)
    def forward(self, x1):
        y = self.a1(torch.relu(x1))
        z = self.a2(y)
        return z
# Inputs to the model
x1 = torch.randn(1, 1, 4, 4)
