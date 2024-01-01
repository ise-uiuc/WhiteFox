
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        y1 = torch.nn.functional.silu(x1).add(42.6)
        return torch.cat([torch.mm(y1, x2), y1], 1)
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
