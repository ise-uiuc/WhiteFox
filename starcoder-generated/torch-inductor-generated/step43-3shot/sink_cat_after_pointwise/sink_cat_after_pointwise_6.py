
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat([x, x], dim=0)
        z = y.view(-1, y.shape[0])
        y = torch.tanh(z)
        return y
# Inputs to the model
x = torch.randn(3, 5)
