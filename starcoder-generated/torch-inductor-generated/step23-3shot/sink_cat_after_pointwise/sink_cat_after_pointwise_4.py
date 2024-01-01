
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat([x[:1], x], dim=1)
        y = y.view(y.shape[0], -1)
        y = y.tanh()
        y = torch.cat([y, y], dim=1)
        x = y.tanh()
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
