
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = x + x
        x = x.view(x.shape[0], -1)
        y = torch.cat((x, x, x), dim=1)
        return y.relu()
# Inputs to the model
x = torch.randn(2, 3, 4)
