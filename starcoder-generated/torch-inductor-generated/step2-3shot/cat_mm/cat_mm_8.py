
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, h):
        r = torch.cat([x, h], 1)
        return r
# Inputs to the model
x = torch.randn(1, 1)
h = torch.randn(1, 1)
