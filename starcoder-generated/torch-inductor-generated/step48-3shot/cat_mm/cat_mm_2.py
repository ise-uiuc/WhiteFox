
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, xl1, xl2):
        v1 = torch.mm(xl1, xl2)
        v2 = torch.mm(xl1, xl2)
        xl1 = xl1.detach()
        xl2 = xl2.detach()
        v3 = torch.mm(xl1, xl2)
        return torch.cat([v1, v2, v3], 1)
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 2)
