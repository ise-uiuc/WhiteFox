
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        x, y = torch.cat((x, y), dim=1), torch.cat((y, x), dim=1)
        x, y = x.relu(), y.abs()
        return x, y
# Inputs to the model
x = torch.randn(2, 3, 4)
y = torch.randn(2, 2, 4)
