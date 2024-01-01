
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x, x), dim=1)
        z1 = y.view(x.shape[0], -1).tanh() if x.shape!= (1, 3) else y.view(x.shape[0], -1).relu()
        z2 = z1.view(y.shape) if (x.shape!= (1, 3) and z1.shape == y.shape) else z1
        return z2.view(y.shape) if y.shape == z1.shape else y.view(x.shape)
# Inputs to the model
x = torch.randn(2, 3, 4)
