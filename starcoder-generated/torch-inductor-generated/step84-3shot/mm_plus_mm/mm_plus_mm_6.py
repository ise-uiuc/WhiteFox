
class Model(torch.nn.Module):
    def forward(self, x, y, z):
        a = torch.mm(x, x)
        b = torch.mm(x, x)
        c = torch.mm(y, z)
        e = torch.mm(x, y)
        return a + b - c + e
# Inputs to the model
x = torch.randn(1, 1)
y = torch.randn(1, 1)
z = torch.randn(1, 1)
