
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, inp1, inp2):
        v1 = torch.mm(inp1, inp1)
        v2 = torch.mm(inp2, inp2)
        v3 = v1 + x
        v4 = v2 + x
        return torch.cat((v3, v4), dim=1)
# Inputs to the model
x = torch.randn(3, 3, requires_grad=True)
inp1 = torch.randn(3, 3)
inp2 = torch.randn(3, 3)
