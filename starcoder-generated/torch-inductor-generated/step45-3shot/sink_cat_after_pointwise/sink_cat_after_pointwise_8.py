
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat([x, x], dim=1)
        x = y.clamp(0).view(y.shape[0], -1)
        x = x.div_(3.14159)
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
