
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        x2 = torch.rand_like(x1)
        x3 = torch.rand_like(x1)
    def forward(self, x1):
        x4 = torch.rand_like(x1)
        return (x3, x4)
# Inputs to the model
x1 = torch.randn(2, 3, 4, 5)
