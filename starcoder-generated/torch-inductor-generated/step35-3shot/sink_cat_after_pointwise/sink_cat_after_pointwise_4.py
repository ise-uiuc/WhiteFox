
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x, x), dim=0)
        y = y.view(y.shape[0], -1).unsqueeze(2)
        y = y.unsqueeze(0)
        x = torch.concat((y, y), dim=0)
        return x
# This model triggers the problem as the concat dimension is wrongly inferred
x = torch.randn(1, 2, 3, 4)
