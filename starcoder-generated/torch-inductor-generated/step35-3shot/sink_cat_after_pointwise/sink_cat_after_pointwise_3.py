
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.arange(40.0)
        x, y = x + y, y + x
        x, y = y.clamp(max=10), x.clamp(max=10)
        z = torch.cat([x, x, y, x, y], dim=0)
        return x
# Inputs to the model
x = torch.arange(10.0)
