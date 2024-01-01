
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        s = torch.unsqueeze(inp, 1)
        r = torch.unsqueeze(x2, 0)
        v1 = torch.squeeze(torch.mm(r, s), 1)
        v2 = v1 + x1
        return v2
# Inputs to the model
x1 = torch.randn(0, 0)
x2 = torch.randn(2, 2)
inp = torch.randn(2, 2)
