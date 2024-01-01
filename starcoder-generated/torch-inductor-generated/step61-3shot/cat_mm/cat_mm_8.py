
class Model(torch.nn.Module):
    def forward(self, x, y, z):
        return torch.cat([torch.mm(x, y)] * 2 + [torch.mm(x, z)] * 3, 1)
# Inputs to the model
x = torch.randn(3, 2)
y = torch.randn(2, 3)
z = torch.randn(2, 2)
