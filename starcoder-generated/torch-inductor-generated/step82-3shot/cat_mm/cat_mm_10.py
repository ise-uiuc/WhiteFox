
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        return torch.cat([torch.mm(x1, x2), torch.mm(x1, x2)], 0)
# Inputs to the model
x1 = torch.randn(16, 8)
x2 = torch.randn(8, 2)
