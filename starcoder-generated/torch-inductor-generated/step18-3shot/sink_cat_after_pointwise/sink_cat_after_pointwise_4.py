
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x), dim=1)
        x = y.view(y.shape[0], -1).tanh() if y.shape[0] == 1 else y.tanh()
        x = x.view(x.shape[0])
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
