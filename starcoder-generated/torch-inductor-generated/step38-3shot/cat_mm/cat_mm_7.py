
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x3, x1)
        return torch.cat([v1, v1, v2, v1, v1, v1, v2], 0)
# Inputs to the model
x1 = torch.randn(1,1)
x2 = torch.randn(1,1)
x3 = torch.randn(1,1)
