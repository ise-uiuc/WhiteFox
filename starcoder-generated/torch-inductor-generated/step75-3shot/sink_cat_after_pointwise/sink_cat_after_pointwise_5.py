
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def g(self, x):
        return x
    def forward(self, x):
        y = self.g(x)
        y = y.view(y.shape[0], -1)
        return y.tanh()
# Inputs to the model
x = torch.randn(2, 3)
