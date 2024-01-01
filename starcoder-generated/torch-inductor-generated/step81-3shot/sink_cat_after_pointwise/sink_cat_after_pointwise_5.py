
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(x.size(0), -1)
        x = torch.cat((x.abs().sum().expand(x.size(0), 1, 1, 1), x), dim=1)
        return x.view(x.size(0), -1)
# Inputs to the model
x = torch.randn(4, 2, 1, 1)
