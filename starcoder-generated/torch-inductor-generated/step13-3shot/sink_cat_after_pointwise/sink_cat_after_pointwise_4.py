
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        z = x.view(x.shape[0], -1)
        y = torch.cat((z, z), dim=1)
        y = x.view(x.shape[0], -1).tanh() if y.shape[0] == 1 else y.tanh()
        return y
# Inputs to the model
x = torch.randn(2, 3, 4)
