
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inp1, inp2, inp3):
        v1 = torch.mm(inp1, inp2)
        v2 = v1 + inp3
        return v2
# Inputs to the model
inp1 = torch.randn(1, 1)
inp2 = torch.randn(1, 1)
inp3 = torch.randn(1, 1)
