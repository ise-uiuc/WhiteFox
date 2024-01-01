
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x = torch.randn(3, 4)
    def forward(self, x):
        y = torch.cat((x, x), dim=1)
        y = y.view(y.shape[0], -1)
        x = self.x
        if x.shape[0]!= y.shape[0]:
            x = x.repeat(y.shape[0], 1, 1)
        y = torch.tanh(y * x)
        return y.sum()
# Inputs to the model
x = torch.randn(2, 3, 4)
