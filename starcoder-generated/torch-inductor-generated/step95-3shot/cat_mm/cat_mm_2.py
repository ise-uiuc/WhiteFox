
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v = torch.mm(x1, x2)
        return torch.cat([v] * 20, 2)
# Inputs to the model
x1 = torch.randn(4, 4)
x2 = torch.randn(3, 4)
