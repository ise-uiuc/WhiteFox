
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        v1 = torch.cat([x, x, x], 0)
        v2 = torch.cat([x, x, x], 1)
        return v1 + v2
# Inputs to the model
x = torch.randn(10, 4)
