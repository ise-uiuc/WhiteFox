
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight1 = torch.randn(3, 3, requires_grad=True)
        self.weight2 = torch.randn(3, 3)
    def forward(self, x3, x4):
        v1 = torch.mm(x3, self.weight1)
        v2 = v1 + self.weight2
        v3 = torch.mm(x4, v2)
        v4 = v1 + torch.mm(x4, self.weight1)
        return v3 + v4
# Inputs to the model
x3 = torch.randn(3, 3)
x4 = torch.randn(3, 3)
