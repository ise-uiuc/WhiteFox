
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        a = torch.randn(1)
        b = torch.randn(1)
        c = torch.randn(1)
        return torch.cat([a, b, c], -1)
# Inputs to the model
x1 = torch.randn(1)
x2 = torch.randn(1)
