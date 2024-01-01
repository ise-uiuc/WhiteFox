
class Model(torch.nn.Module):
    def forward(self, x, y, z):
        if x is None: pass
        return torch.add(y, z)
# Inputs to the model
x = torch.randn(1, 3, 4)
y = torch.randn(1, 5, 6)
z = torch.randn(x.shape[0], y.shape[1])
