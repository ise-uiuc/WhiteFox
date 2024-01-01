
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x / 2
        y = torch.cat([x, x, x], dim=1)
        y = y.view(y.shape[0], -1)
        y = y.clamp(min=0)
        z = y / 2
        z = torch.cat([z, y], dim=1)
        z = z.view(x.shape[0], -1)
        z = torch.relu(z)
        return z
# Inputs to the model
x = torch.randn(2, 3, 4)
