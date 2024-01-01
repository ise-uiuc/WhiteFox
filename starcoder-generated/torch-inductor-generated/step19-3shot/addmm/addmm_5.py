
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = torch.mm(inp, inp)
        v2 = v1.sum()
        v3 = v1.mean(dim=1, keepdim=True)
        v4 = v3.view(x1.size(0), -1)
        return torch.mm(v4, inp)
# Inputs to the model
x1 = torch.randn(5, 5)
x2 = torch.randn(5, 5)
inp = torch.randn(5, 5)
