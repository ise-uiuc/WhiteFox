
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        z = x * x
        z = z + z
        z = z.tanh()
        print(z)
        z = x * y
        z = z + y
        z = z.relu()
        return z
# Inputs to the model
x = torch.randn(1, 2)
y = torch.randn(1, 3)
