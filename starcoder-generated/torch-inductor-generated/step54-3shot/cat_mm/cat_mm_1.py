
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x1 + x2
        v2 = x1 * x2
        return v1, v2
# Inputs to the model
x1 = torch.randn(1, 15)
x2 = torch.randn(15, 1)
x3 = torch.randn(15, 1)
x4 = torch.randn(15, 1)
