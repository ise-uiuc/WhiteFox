
class Model(torch.nn.Module):
    def forward(self, x, y, z1, z2):
        a = torch.mm(x, z1)
        b = torch.mm(z2, y)
        c = a + b
        return c
# Inputs to the model
x = torch.randn(32, 3)
y = torch.randn(32, 3)
z1 = torch.randn(32, 3)
z2 = torch.randn(32, 3)
