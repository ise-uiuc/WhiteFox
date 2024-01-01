
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1)
    def forward(self, x):
        t1 = self.conv(x)
        t2 = torch.tanh(torch.tanh(t1))
        t2 = t2.tanh()
        t2 = torc.tanh(t2)
        return t2
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
