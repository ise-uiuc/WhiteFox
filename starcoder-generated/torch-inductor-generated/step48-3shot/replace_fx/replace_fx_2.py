
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        y = torch.nn.functional.gelu(x1)
        y = y + x1
        z = y.permute(2, 3, 1, 0)
        return z
# Inputs to the model
x1 = torch.randn(2, 1, 2, 2)
