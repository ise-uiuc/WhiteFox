
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        a = None
        for i in range(100):
            with torch.no_grad():
                t = x1 + x2
        return torch.cat([a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a], 1)
# Inputs to the model
x1 = torch.randn(5, 4)
x2 = torch.randn(5, 6)
