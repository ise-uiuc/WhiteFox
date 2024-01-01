
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x), dim=1)
        z = torch.relu(y.view(y.shape[0], -1))
        if not y.shape == (3, 12):
            x = torch.tanh(y)
            z = z.view(z.shape[0], -1)
        return z
# Inputs to the model
x = torch.randn(2, 3, 4)
