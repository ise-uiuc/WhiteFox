
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x + 1
        x = torch.cat((x, y), dim=1).view(x.shape[0], -1)
        y = x.view(-1)
        x = y.sigmoid() * torch.cat((x, y), dim=1)
        return x
# Inputs to the model
x = torch.randn(2, 2, 2)
