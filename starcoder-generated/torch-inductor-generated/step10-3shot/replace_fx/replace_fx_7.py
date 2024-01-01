
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        a1 = torch.randn(1)
        a2 = torch.randn(1)
        a3 = a2 + x1
        a4 = a3 + a1
        return a1 * a4 + a4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
