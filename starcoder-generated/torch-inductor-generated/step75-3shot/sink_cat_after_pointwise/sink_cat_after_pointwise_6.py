
class Model(torch.nn.Module):
    def __init__(self):
        super(). __ init__()
        self.module = torch.nn.Linear(2, 5)

    def forward(self, x):
        y = torch.cat([x, x], dim=1)
        y = self.module(y)
        y = torch.cat([x, y], dim=1)
        return y.tanh()
# Inputs to the model
x = torch.randn(2, 2)
