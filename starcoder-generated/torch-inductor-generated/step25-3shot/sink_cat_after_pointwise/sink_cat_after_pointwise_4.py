
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.add(1)
        y = torch.cat((x, x, x), dim=1)
        y = y.view(y.shape[0], -1)
        if y.dim() == 2:
            z = torch.relu(y)
            z = z.tanh()
            z = z.repeat(5, 1, 1)
            return z
        else:
            return x
# Inputs to the model
x = torch.randn(2, 3, 4)
