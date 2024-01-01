
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = x1 * x1
        v1 = torch.rand_like(x2)
        return v1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
