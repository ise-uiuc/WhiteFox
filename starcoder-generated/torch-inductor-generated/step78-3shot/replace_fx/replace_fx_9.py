
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.rand_like(x1)
        x3 = torch.rand_like(x3)
        x4 = torch.rand_like(x2, dtype=torch.float32)
        return (x2, x4)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
