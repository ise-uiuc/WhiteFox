
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mm1 = torch.nn.Linear(4, 4)
    def forward(self, x, y, z):
        self.mm1.weight = torch.Tensor([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0], [4.0, 4.0, 4.0, 4.0]])
        v1 = self.mm1(x)
        v2 = torch.mm(y, v1)
        v3 = v2 + z
        return v3
# Inputs to the model
x = torch.randn(3, 4)
y = torch.randn(3, 4)
z = torch.randn(3, 4)
