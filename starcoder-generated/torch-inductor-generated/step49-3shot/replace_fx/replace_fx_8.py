
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.rand(100)
        for _ in range(20):
            x *= 2
        return x
# Inputs to the model
x1 = torch.randn(1)
x2 = torch.randn(100)
