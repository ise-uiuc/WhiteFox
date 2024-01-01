
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t0, t1 = torch.split(x, 1, dim=0)
        return torch.cat([x, t0], dim=0).tanh()
# Inputs to the model
x = torch.randn(2, 3, 4)
