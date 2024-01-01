
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.tanh(x.view(-1))
        return torch.cat([x, x], dim=1).view(x.shape[0], -1).tanh()
# Inputs to the model
x = torch.randn(4, 2, 3)
