
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x), dim=1)
        x = y.view(-1) if (x.shape[0], 2*x.shape[1]) == (1, 6) else y.view(x.shape[0], -1)
        x = x.tanh() if x.shape == (1, 2) else x.tanh()
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
