
class Model(torch.nn.Module):
    def forward(self, x, y, z):
        a = torch.mm(x, x)
        b = torch.mm(x, x)
        c = torch.mm(x, y)
        d = torch.mm(y, x)
        return x + a + b + c + d
# Inputs to the model
x = torch.randn(10, 10)
y = torch.randn(10, 10)
z = torch.randn(10, 10)
