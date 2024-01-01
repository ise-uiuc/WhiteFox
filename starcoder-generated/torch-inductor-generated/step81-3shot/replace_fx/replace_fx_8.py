
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x1 = x1 + 1.0 # Make tensor larger in size
        x2 = torch.rand_like(x1)
        return x2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
