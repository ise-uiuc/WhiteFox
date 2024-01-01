
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        return torch.cat([torch.cat([torch.mm(x1, x2), torch.mm(x1, x2), torch.mm(x1, x2), torch.mm(x1, x2)], 1) for i in range(3)], 0)
# Inputs to the model
x1 = torch.randn(6, 3)
x2 = torch.randn(3, 1)
