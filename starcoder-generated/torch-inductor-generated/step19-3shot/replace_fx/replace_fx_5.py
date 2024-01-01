
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x1 = torch.rand_like(x, dtype=torch.float)
        x2 = torch.rand_like(x, dtype=torch.float)
        x3 = torch.rand_like(x, dtype=torch.float)
        x4 = torch.rand_like(x, dtype=torch.float)
        x5 = torch.rand_like(x, dtype=torch.float)
        return x1 + x2
# Inputs to the model
x = torch.rand(3, 2, 2)
