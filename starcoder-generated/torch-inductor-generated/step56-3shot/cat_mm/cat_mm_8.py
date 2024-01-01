
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.cat([torch.mm(x1, x2), torch.mm(x1, x2)], 1)
        return torch.cat([torch.mm(x1, x2), v1, torch.mm(x1, x2)], 1)
# Inputs to the model
x1 = torch.randn(3, 4)
x2 = torch.randn(3, 3)
