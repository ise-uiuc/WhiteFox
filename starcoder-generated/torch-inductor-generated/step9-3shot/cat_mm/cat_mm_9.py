
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x1, x2)
        v3 = torch.nn.functional.tanh(v1)
        v4 = torch.nn.functional.hardtanh(v2)
        v5 = torch.cat([v3, v4], 0)
        return v5
# Inputs to the model
x1 = torch.randn(1, 1)
x2 = torch.randn(1, 1)
