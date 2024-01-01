
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x), dim=1)
        z = y.view(y.shape[0], -1)
        x = z.tan() if z.shape[0] == 1 else z.tan()
        x = x.relu()
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
