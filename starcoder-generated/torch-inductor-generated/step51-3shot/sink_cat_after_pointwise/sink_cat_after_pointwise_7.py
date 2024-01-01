
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x), dim=1)
        x = y.view(y.shape[0], -1)
        y = x.tanh()
        return torch.sin(y).sum(dim=-1)
# Inputs to the model
x = torch.randn(2, 3, 4)
