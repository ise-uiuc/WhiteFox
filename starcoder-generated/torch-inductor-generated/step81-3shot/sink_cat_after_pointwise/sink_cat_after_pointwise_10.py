
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(x.shape[0] * x.shape[1], *x.shape[2:]) if x.shape[1] > 1 else x.view(x.shape[0], -1)
        x = torch.cat((y, y), dim=1) if y.shape == (3, 4) or y.shape == (3, 8) else y.relu()
        return torch.tanh(x)
# Inputs to the model
x = torch.randn(3, 2, 2)
