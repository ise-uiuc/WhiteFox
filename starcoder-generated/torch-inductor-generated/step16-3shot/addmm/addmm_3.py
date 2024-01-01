
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x2):
        t = torch.randn(1,0)
        v1 = torch.mm(x2 + t, t)
        v2 = torch.mm(v1, t)
        v3 = torch.mm(v1, x2)
        v4 = v3 + v2
        return v4

# Inputs to the model
x2 = torch.randn(1,1)

