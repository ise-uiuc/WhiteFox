
class Model(torch.nn.Module):
    def forward(self, x, y, z):
        a = torch.mm(x, z)
        b = torch.mm(y, z)
        return a + b
# Inputs to the model
x = torch.randn(5, 5)
y = torch.randn(5, 5)
z = torch.randn(5, 5)
