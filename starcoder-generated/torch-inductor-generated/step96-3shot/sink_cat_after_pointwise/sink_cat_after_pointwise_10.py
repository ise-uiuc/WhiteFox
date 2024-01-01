
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat([x, x], dim=1)
        while y.shape[1] < x.shape[2]:
            y = torch.cat([y, x], dim=1)
        y = torch.cat([y, x], dim=1)
        y = y.view(y.shape[0], -1)
        y = torch.tanh(y)
        return torch.relu(y)
# Inputs to the model
x = torch.randn(2, 2, 2)
