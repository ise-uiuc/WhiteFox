
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(x.shape[0], -1)
        y = y.view(y.shape[0], -1).tanh()
        x1, x2 = torch.chunk(y, 2, dim = 0)
        y = torch.cat((x1, x2), dim=1)
        y = y.view(-1).tanh()
        x = x.tan()
        y = y.view(-1).tanh()
        x = torch.cat((x, y), dim=0)
        x = x.tanh()
        x = x.tanh()
        return x
# Inputs to the model
x = torch.randn(6, 3, 4)
