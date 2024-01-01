
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = Model2()
    def forward(self, x):
        z = torch.cat([x, x], dim=1)
        x = self.inner(z).tanh()
        return x

class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(x.shape[0], -1)
        return y
# Inputs to the model
x = torch.randn(2, 3, 4)
