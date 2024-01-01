
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(16, 64, 3, 1, 1)
    def forward(self, x):
        t = torch.tanh(self.conv(x))
        y = torch.tanh(t)
        return y
# Inputs to the model
x = torch.randn(1, 16, 112, 112)
