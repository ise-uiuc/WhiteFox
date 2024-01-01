
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        y = x.view(x.shape[0], -1)
        y = torch.cat((y, y), dim=1).sigmoid() if y.shape[0] == 1 else torch.cat((y, y), dim=1).sigmoid()
        x = y + x.sum(dim=1).sigmoid()
        return x
# Inputs to the model
x = torch.randn(2, 2)
y = torch.randn(2, 2)
