
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        r = torch.randn(3, 3)
    def forward(self, x1, x2, inp):
        v1 = torch.mm(inp, x1)
        v2 = v1 + x1
        x2.squeeze(0)
        v1 = v1 + v1
        x1.unsqueeze(0)
        inp = x1
        return v2
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3, requires_grad=True)
inp = torch.randn(3, 3, requires_grad=True)
