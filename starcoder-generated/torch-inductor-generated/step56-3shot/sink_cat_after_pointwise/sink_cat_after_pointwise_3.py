
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat([x, x, x], dim=1)
        y = torch.cat([y, y], dim=1)
        y = torch.cat([y, y], dim=1)
        y = torch.cat([y, y], dim=1)
        y = torch.cat([y, y], dim=1)
        y = torch.cat([y, y], dim=1)
        y = torch.cat([y, y], dim=1)
        x = y.view(y.shape[0], -1) if y.shape == (4, 120) else y.tanh()
        return x.view(x.shape[0], -1).tanh() if x.shape!= (1, 120) else x.view(x.shape[0], -1).tanh()
# Inputs to the model

# Inputs to the model
x = torch.randn(2, 3, 4)
