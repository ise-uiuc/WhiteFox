
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, x1):
        x = x.view(x.shape[0], -1)
        x = torch.cat((x, x1), dim=1)
        x = x.view(x.shape[0], 2, -1)
        y = torch.tanh(x)
        return y if y.shape!= (1,) else y
# Inputs to the model
x = torch.randn(2, 1, 8)
x1 = torch.randn(2, 1, 8)
