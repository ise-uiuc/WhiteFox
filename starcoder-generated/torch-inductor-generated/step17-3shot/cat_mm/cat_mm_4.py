
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        return torch.cat([torch.cat([torch.mm(i1, i2), torch.mm(i1, i2)], 1) for i1, i2 in zip(x1, x2)], 1)
# Inputs to the model
x1 = torch.randn(2, 3)
x2 = torch.randn(3, 1)
