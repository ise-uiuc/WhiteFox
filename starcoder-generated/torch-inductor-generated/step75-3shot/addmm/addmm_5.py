
# Add your code here
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y, z):
        inp = torch.mm(y, z)
        return torch.mm(z, x) + inp
# Inputs to the model:
x = torch.randn(2, 2)
y = torch.randn(2, 2)
z = torch.randn(2, 2)
