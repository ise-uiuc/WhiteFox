
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x1 = torch.rand_like(x, dtype=torch.long)
        x2 = torch.relu6(x)
        return x2
# Inputs to the model
x1 = torch.randn(1, 6)
