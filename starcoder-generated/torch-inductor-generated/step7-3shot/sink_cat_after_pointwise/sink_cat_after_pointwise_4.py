
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        v1 = torch.cat((x.view(x.shape[0], -1), y.view(y.shape[0], -1)), dim=1)
        x = x.reshape(-1)
        y = y.view(-1)
        v2 = v1.tanh()
        x = x.tanh()
        y = y.tanh()
        v3 = v2 + x + y
        return v3
# Inputs to the model
x = torch.randn(2, 3)
y = torch.randn(3, 2)
