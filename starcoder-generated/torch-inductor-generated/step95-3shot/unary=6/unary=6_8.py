
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(1)
    def forward(self, x1):
        t1 = x1.reshape(x1.size()[0], x1.size()[1] * 3, 28, 28)
        t2 = self.bn(t1)
        return  x1.reshape(x1.size()[0], x1.size()[1], 28, 28)
# Inputs to the model
x1 = torch.randn(1, 3, 112, 112)
