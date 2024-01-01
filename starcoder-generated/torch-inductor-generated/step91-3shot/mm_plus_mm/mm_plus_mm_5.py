
class Model(torch.nn.Module):
    def forward(self, x, y):
        x = x - y
        y = x ** 2
        z = y - x
        w = z * y
        z = y * x
        z = z + w
        return z
# Inputs to the model
x = torch.randn(4, 4)
y = torch.randn(4, 4)
