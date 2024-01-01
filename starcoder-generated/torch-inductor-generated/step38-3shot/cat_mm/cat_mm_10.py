
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.torch.nn.functional.tanh(torch.mm(x1, x2))
        v2 = torch.mm(x1, x2).pow(2)
        return torch.cat([v2, v2, v2, v1, v2, v1, v2, v1, v1], 1)
# Inputs to the model
x1 = torch.randn(1, 2)
x2 = torch.randn(2, 1)
