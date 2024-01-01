
class Model(torch.nn.Module):
    def __init__(self, p1):
        super().__init__()
        self.p1 = p1
    def forward(self, x1):
        x2 = x1 + self.p1
        x3 = torch.rand_like(x2)
        return x3
p1 = torch.tensor([2.0])
# Inputs to the model
x1 = torch.randn(6, 6)
