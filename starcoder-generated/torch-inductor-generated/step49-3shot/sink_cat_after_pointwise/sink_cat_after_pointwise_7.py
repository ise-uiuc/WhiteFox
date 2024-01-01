
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x,x), dim=1)
        y = y.view(-1)
        x = y.tanh() if (x.shape[1] == 2 * x.shape[1] or (x.shape[0], 2 * x.shape[0]) == (3, 6) or (x.shape[0], 2 * x.shape[0]) == (3, 6)) else y
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
