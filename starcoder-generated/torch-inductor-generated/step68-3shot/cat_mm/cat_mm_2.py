
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.cat([torch.mm(x, x) for _ in range(10)], 1)
# Input to the model
x = torch.randn(1, 3)
