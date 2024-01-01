
class Model(torch.nn.Module):
    def __init__(self):
         super().__init__()
    def forward(self, x):
        y = torch.cat([x, x, x, x], dim=1)
        z = y.view(y.shape[0], -1)
        w = torch.relu(z)
        return w.tanh().view(y.shape[0], -1)
# Inputs to the model
x = torch.randn(2, 3, 4)
