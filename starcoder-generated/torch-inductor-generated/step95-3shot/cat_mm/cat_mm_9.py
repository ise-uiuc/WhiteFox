
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v = torch.mm(x1, x2)
        v = torch.cat(2, [v] * 20)
        return torch.cat([v] * 20, 1)
# Inputs to the model
x1 = torch.randn(2, 4)
x2 = torch.randn(4, 4)
