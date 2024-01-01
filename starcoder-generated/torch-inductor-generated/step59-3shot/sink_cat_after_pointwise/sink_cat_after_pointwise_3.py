
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x1 = torch.sin(x)
        x2 = x.repeat(10, 1, 1)
        y = torch.cat((x1, x2), dim=1)
        w = torch.relu(y)
        z = torch.tanh(y)
        return z.view(z.shape[0], -1)
# Inputs to the model
x = torch.randn(2, 3, 4)
