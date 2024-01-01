
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat([x, x, x], dim=0)
        z = y.view(y.shape[0], -1)
        return z.tanh() if y.shape!= (6, 9) else z.view(z.shape[0], z.shape[1], -1).sigmoid()
# Inputs to the model
x = torch.randn(2, 3, 4)
