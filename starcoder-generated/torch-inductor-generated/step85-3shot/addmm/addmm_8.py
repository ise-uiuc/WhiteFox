
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inp1, inp2, inp3, x1, x2, x3):
        v1 = torch.mm(inp1, x1) + inp2
        v2 = torch.mm(inp3, x2) - inp3
        return v1/v2 # divide the result of v1 and v2
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3, requires_grad=True)
x3 = torch.randn(3, 3)
inp1 = torch.randn(3, 3)
inp2 = torch.randn(3, 3)
inp3 = torch.randn(3, 3)
