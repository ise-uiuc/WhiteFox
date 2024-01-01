
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = x1 + x1
        x3 = x1 + torch.rand_like(x2)
        x4 = x1 + x1
        x5 = x1 + x1
        return (x5, x4)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
