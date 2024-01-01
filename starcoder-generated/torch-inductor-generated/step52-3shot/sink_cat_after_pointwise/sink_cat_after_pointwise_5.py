
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.mean(axis=2)
        y = y.view(y.shape[0], -1)
        y = x.tanh()
        y = torch.cat((y, y), dim=1)
        x = y.mean(axis=2)
        return x
# Inputs to the model
x = torch.randn(2, 2, 2)
