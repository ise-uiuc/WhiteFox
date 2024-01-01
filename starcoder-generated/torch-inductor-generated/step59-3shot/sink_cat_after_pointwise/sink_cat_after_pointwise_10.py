
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, a, b, c, d, e):
        return torch.cat((a, b, c, d, e), dim=3).view(5, -1).relu()
# Inputs to the model
a = torch.randn(3, 4)
b = torch.randn(3, 4)
c = torch.randn(3, 4)
d = torch.randn(3, 4)
e = torch.randn(3, 4)
