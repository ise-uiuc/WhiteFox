
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        x20 = torch.rand_like(x1)
        o1 = torch.add(x1, x1)
        o2 = o1 * x2
        o2 = o2 / (o2 * x2)
        o2 = o2 + x20
        return o2
# Inputs to the model
x1 = torch.randn(10)
x2 = torch.randn(10)
