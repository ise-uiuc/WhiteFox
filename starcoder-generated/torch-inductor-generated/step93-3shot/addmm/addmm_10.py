
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        x1 = x1 + torch.mm(inp, x2)
        v1 = np.exp(x1)
        v1 = torch.add(torch.mm(v1, x1), v1)
        return v1
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
inp = torch.randn(3, 3, requires_grad=True)
