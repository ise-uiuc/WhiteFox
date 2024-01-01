
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        y = (y.unsqueeze(-2) * y.unsqueeze(-1)).view(y.shape[0], -1)
        x = y.cat((x, y), dim=0)
        return torch.add(x, y)
# Inputs to the model
x = torch.randn(1, 2)
y = torch.randn(1, 2)
