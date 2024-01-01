
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, a, b, c, d):
        x = torch.cat([a, b, c, d], dim=0)
        y = x[:, 0]
        return y
# Inputs to the model
a = torch.randn(2, 2)
b = torch.randn(3, 2)
c = torch.randn(4, 2)
d = torch.randn(5, 2)
