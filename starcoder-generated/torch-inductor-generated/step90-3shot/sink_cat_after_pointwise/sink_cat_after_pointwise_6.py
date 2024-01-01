
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat([x, x, x], dim=1)
        y = y.view(x.shape[0], -1).sigmoid()
        y = y.tanh()
        return y
# Inputs to the model
x = torch.randn((3, 4), requires_grad=True)
