
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x, x), dim=1)
        y = y.view(y.shape[0], y.shape[1], -1).tanh()
        return y if y.shape!= (1, 4, 1) else y.view(0, 0, -1)
# Inputs to the model
x = torch.randn(2, 3, 4)
