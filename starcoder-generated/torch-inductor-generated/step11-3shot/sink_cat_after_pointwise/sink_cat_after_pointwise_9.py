
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(x.shape[0], -1)
        y.tanh()
        y = torch.cat((y, y), dim=1)
        x = y.view(y.shape[0], -1)
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
