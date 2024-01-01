
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh1 = torch.nn.Tanh()
        self.view1 = torch.nn.Unflatten(1, (20, 2, 2))
    def forward(self, x):
        y = self.tanh1(x)
        y = self.view1(y)
        x = self.tanh1(y)
        return x
# Inputs to the model
x = torch.randn(2, 20, 1, 1)
