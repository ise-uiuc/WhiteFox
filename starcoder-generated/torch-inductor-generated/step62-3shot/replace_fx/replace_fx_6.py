
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        a1 = torch.rand(4096, 4096, requires_grad=True)
        a2 = a1[:, 1]!= 0
        b = torch.rand(4096, requires_grad=True)
        c = b[a2]
        d = torch.rand(sum(a2), requires_grad=True)
        return d
# Inputs to the model
x1 = torch.randn(1, 2, 2)
