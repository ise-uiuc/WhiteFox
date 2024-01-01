
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp, x5):
        v1 = torch.mm(x1, x5)
        v2 = np.tanh(v1)
        v3 = torch.mm(inp, v2)
        return v3 + x1
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
x3 = torch.randn(1, 3, requires_grad=True)
x4 = torch.zeros(1, 3)
inp = torch.randn(3, 3)
x5 = torch.randn(1, 3, requires_grad=True)
