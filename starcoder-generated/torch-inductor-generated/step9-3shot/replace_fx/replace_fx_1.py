
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        a1 = rand_like(x1, 0.1, True, torch.float64)
        a2 = rand_like(x1, 0.1, True, torch.float64)
        a3 = x1 + x1
        a4 = a3 + a2
        return a4
# Inputs to the model
x1 = torch.randn(3, 3)
