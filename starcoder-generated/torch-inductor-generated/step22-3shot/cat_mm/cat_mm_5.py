
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        l1 = torch.mm(x1, x2)
        l2 = torch.mm(x1, x2)
        return torch.cat([l1, l2], 0)
# Inputs to the model
x1 = torch.randn(3, 4)
x2 = torch.randn(4, 5)
