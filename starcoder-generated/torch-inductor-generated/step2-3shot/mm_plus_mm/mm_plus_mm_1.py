
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, x4):
        v1 = torch.mm(x1, x4)
        v2 = torch.mm(x2, x3)
        v3 = v1 + v2
        return v3
print(x1)
print(x2)
# Inputs to the model
x1 = torch.randn(3, 5)
x2 = torch.randn(5, 3)
x3 = torch.randn(3, 5)
x4 = torch.randn(5, 3)
