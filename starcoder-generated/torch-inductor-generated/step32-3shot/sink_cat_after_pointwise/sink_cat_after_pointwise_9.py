
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat(x, x, x)
        z = y.view(x.shape[0], -1).tanh() if torch.numel(y) == torch.numel(x) else y.view(x.shape[0], -1).tanh()
        return z


# Inputs to the model
x = torch.randn(2, 3, 4)
