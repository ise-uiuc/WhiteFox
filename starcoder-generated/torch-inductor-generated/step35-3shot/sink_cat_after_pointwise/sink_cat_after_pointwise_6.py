
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        z = torch.cat((x, x), dim=1)
        if z.shape[0] == 1: z.add(z)
        a = (z.mean(), y.max())
        b = a[0] + a[1]
        a = x if a[0] == -1 else torch.relu(y)
        return b * a
# Inputs to the model
x = torch.randn(2, 3, 4)
y = torch.randn(3, 5, 6)
