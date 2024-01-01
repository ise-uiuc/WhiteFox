
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x), dim=1)
        y.view(x.shape[0], -1)
        y = y.tanh()
        x = y.view(-1, 2)
        y = y.relu() if x.shape[0] == 1 else y
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
