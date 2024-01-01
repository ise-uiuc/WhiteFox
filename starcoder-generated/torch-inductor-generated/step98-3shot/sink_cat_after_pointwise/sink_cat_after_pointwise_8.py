
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        z = torch.cat([x, x], dim=1)
        return z
# Inputs to the model
x = torch.randn(2, 3, 4)
