
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 3, False)
        self.linear2 = torch.nn.Linear(3, 3, False)
    def forward(self, x1, x2, x3, x4, x5):
        v1 = self.linear1(x1)
        v2 = x5 + x3
        v3 = self.linear2(v1)
        v4 = v2 + v3
        return v4
# Inputs to the model
x1 = torch.randn(3, 2)
x2 = torch.randn(3, 2)
x3 = torch.randn(3, 2)
x4 = torch.randn(3, 2)
x5 = torch.randn(3, 2, requires_grad=True)
