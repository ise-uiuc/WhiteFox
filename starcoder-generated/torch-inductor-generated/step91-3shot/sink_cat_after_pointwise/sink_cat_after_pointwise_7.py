
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.ones(3, requires_grad=True)
        z = torch.cat((x, x), dim=1)
        if z.shape == (2, 10):
            return z.tanh()
        return y.tanh()
# Inputs to the model
x = torch.randn(1, 2, 2)
