
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.randn(4, 6)
    def forward(self, x):
        z = self.weight.view(4, -1)
        x = x.view(1, -1)
        w = self.weight.view(-1, 6)
        y = torch.dot(x, w)
        y = y.tanh()
        return y
# Inputs to the model
x = torch.randn(1, 4, 30, 30)
