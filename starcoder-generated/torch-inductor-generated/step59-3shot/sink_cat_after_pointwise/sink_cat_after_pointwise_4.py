
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x, x), dim=1)
        y = y.view(y.shape[0], -1).tanh() if y.shape!= (1, 3) else y.view(y.shape[0], -1).relu()
        return y[0, :]
# Inputs to the model
x = torch.randn(2, 3, 4)
