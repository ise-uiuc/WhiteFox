
class Model(torch.nn.Module):
    def forward(self, x, y, z, t):
        a = torch.mm(x, z)
        b = torch.mm(z, t)
        c = torch.mm(y, t)
        return a + b + c
# Inputs to the model
x = torch.randn(7, 7)
y = torch.randn(7, 7)
z = torch.randn(7, 7)
t = torch.randn(7, 7)
