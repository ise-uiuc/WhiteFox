
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.sqrt(x1)
        x3 = x2 * x2
        x4 = x3 * x3
        x5 = 0.0 * x3 + x4
        x6 = x5 * x4
        x7 = x6 * x6
        return x5 + x7
# Inputs to the model
x1 = torch.randn(1, 2, 2)
