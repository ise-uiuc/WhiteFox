
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x), dim=1)
        y = y.view(y.shape[0], -1)
        y = y.tanh()
        x = y if y.shape[0] == 1 else y.tanh()
        return x.view(-1)
# Inputs to the model
x = torch.randn(2, 3, 4)
