
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v0 = torch.mm(x1, x2)
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x1, x2)
        v3 = v0*0.5 + v1*0.5
        v4 = v0*0.3 + v1*0.7
        return torch.cat([v2, v3, v4, v2], 1)
# Inputs to the model
x1 = torch.randn(1, 1)
x2 = torch.randn(1, 1)
