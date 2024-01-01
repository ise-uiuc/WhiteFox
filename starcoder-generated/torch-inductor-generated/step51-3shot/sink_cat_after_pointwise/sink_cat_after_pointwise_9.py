
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        a = torch.cat((x, x), dim=1)
        y = a.view(a.shape[0], -1)
        a = a.tanh()
        z = torch.cat((y, y), dim=-2)
        return z + y.sum()
# Inputs to the model
x = torch.randn(2, 3, 4)
