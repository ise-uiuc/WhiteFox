
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, x4, y):
        v1 = torch.mm(x1, y) + x2
        v2 = torch.mm(x3, y) + x4
        m1 = torch.mm(v1, v2) # Perform matrix multiplication
        return m1
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
x3 = torch.randn(3, 3)
x4 = torch.randn(3, 3)
y = torch.randn(3, 3, requires_grad=True)
