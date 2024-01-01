
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp1, inp2):
        v1 = torch.mm(inp1, x2)
        v2 = torch.mm(x1, x2)
        return v1 + v2
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3, requires_grad=True)
inp1 = torch.randn(3, 3)
inp2 = torch.randn(3, 3)
