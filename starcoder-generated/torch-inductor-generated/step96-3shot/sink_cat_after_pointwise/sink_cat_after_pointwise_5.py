
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat([x, x, x], dim=1)
        z = torch.cat([x, x], dim=2)
        x = y.view(y.shape[0], -1)
        x = x.tanh()
        x = z.view(x.shape[0], -1)
        x = torch.relu(x)
        return x
# Inputs to the model
x = torch.randn(2, 2, 3)
