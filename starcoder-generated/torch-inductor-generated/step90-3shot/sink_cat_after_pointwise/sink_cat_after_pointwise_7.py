
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        x = x.transpose(-2, -1)
        y = y.permute((-2, -1))
        x = torch.cat([x, y, y, x], dim=1)
        x = x.view(-1, 8, 4 * 3)
        return x.permute((-2, 2, -1))
# Inputs to the model
x = torch.randn(2, 4, 5)
y = torch.randn(2, 3, 5)
