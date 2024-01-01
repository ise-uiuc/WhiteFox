
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x), dim=-1).view(x.shape[0], -1)
        return y.tanh() if y.shape == (1, 2) else y.tanh()
# Inputs to the model
x = torch.randn(2, 3, 4)
