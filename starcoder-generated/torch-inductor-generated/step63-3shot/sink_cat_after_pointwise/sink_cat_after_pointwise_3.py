
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.cat((x, x), dim=2).view(x.shape[0], -1, 1, 1)
        return x.sum()
# Inputs to the model
x = torch.randn(2, 3, 4, 6)
