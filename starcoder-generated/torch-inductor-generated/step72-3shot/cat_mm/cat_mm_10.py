
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.randn(2, 1)
    def forward(self, x1, x2):
        v1 = torch.mm(x1, self.weight)
        v2 = torch.mm(x2, self.weight)
        v3 = 3 * v1
        v4 = 0.5 * v2
        return torch.cat((v1, v2, v3, v4), 1)
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 2)
