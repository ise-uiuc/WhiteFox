
class Model(torch.nn.Module):
    def forward(self, x, y, z):
        a = torch.mm(x, y)
        b = torch.mm(b, y)
        out = torch.mm(y, z)
        return out
# Inputs to the model
x = torch.randn(64, 64)
y = torch.randn(64, 64)
z = torch.randn(64, 64)
