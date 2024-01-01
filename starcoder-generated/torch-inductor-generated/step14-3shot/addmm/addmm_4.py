
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp1, inp2, inp3, inp4):
        v1 = torch.mm(inp1, inp2)
        v2 = v1 + x1
        v3 = torch.mm(v2, inp3)
        v4 = v3 + inp4
        return v4
# Inputs to the model
x1 = torch.randn(3, 78)
x2 = torch.randn(3, 78)
inp1 = torch.zeros(1, 78)
inp2 = torch.zeros(1, 78)
inp3 = torch.zeros(3, 78)
inp4 = torch.zeros(1, 3)
