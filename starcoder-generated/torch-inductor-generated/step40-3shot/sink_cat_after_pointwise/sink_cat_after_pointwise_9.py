
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.cat([x, x], dim=1).view(x.shape[0], -1).tanh()
# Inputs to the model
x = torch.randn(32, 32, 3)
