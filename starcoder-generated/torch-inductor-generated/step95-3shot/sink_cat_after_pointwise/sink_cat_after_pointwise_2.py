
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        v1 = torch.cat((x.tanh(), x.tanh(), x.tanh()), dim=1)
        v2 = torch.cat((y.tanh(), y.tanh(), y.tanh()), dim=1)
        z = v1 + v2
        return z.view(z.shape[0], -1).relu() if z.shape!= (1, 3) else z.view(z.shape[0], -1).tanh()
# Inputs to the model
x = torch.randn(2, 3, 4)
y = torch.randn(2, 3, 4)
