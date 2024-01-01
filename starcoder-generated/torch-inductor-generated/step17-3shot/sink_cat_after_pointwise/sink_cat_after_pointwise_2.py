
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(x.shape[0], -1)
        y[:, :3] = -y[:, :3]
        y = y.relu().tanh()
        y = torch.cat((x, y), dim=1)
        z = torch.tanh(torch.tanh(y))
        return z
# Inputs to the model
x = torch.randn(2, 3, 4)
