
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(x.shape[0], -1)
        y = y.tanh() if y.shape == (2, 4) or y.shape == (2, 8) else y.relu()
        y = torch.cat((y, y), dim=1)
        x = y.view(y.shape[0], -1).tanh()
        return x
# Inputs to the model
x = torch.randn(2, 2, 2)
