
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        a1 = torch.rand_like(x1)
        a2 = torch.rand_like(x1)
        a3 = torch.rand_like(x1)
        c1 = torch.rand_like(x1)
        c2 = torch.rand_like(x1)
        c3 = torch.rand_like(x1)
        return torch.rand_like(x1)
# Inputs to the model
x1 = torch.randn(1)
