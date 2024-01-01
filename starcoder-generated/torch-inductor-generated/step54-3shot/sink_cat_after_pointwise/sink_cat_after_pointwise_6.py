
x = torch.randn(3, 2, 2)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.arange(3)
        x = torch.cat((x, x), dim=1)
        x = x.view(x.shape[0], -1)
        return x.tanh()
# Inputs to the model
x = torch.randn(3, 2, 2)
