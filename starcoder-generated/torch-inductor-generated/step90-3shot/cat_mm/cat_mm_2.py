
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        return torch.cat([torch.mm(x2, x1), torch.mm(x1, x2), torch.mm(x1, x2)], 1)
# Inputs to the model
x1 = torch.randn(2, 4)
x2 = torch.randn(4, 3)
