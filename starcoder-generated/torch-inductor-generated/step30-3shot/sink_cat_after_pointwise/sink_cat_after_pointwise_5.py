
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.flatten(0, 1)
        y = torch.cat([x, x], dim=1)
        return y
# Inputs to the model
x = torch.randn(2, 3, 4)
