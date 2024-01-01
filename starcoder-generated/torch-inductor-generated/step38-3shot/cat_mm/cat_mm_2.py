
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        return torch.cat([torch.mm(x1[0, :], x2), torch.mm(x1[1, :], x2), torch.mm(x1[2, :], x2)], 1)
# Inputs to the model
x1 = torch.randn(3, 5)
x2 = torch.randn(5, 1)
