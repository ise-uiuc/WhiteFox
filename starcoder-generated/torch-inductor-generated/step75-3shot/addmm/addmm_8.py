
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(12, 1)
    def forward(self, x1, x2, x3, x4, x5, x6):
        v1 = self.linear1(torch.cat([x1.reshape(1, -1), x2.reshape(1, -1), x3, x4, x5, x6], dim=1))
        return v1
# Inputs to the model
x1 = torch.randn(12)
x2 = torch.randn(12)
x3 = torch.randn(3, 3, requires_grad=False)
x4 = torch.randn(3, 3, requires_grad=False)
x5 = torch.randn(3, 3, requires_grad=True)
x6 = torch.randn(3, 3, requires_grad=True)
