
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        x = torch.cat((x, x), dim=1)
        x = torch.nn.functional.linear(x, y)

        y = torch.cat([x, x], dim=1)
        y = torch.nn.functional.linear(y, x)

        z = torch.cat([x, y], dim=0)
        z = torch.nn.functional.linear(z, x)

        return z

# Inputs to the model
x = torch.randn(32, 3, requires_grad=True)
y = torch.randn(3, 32, requires_grad=True)
