
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        return torch.cat([torch.add(x1, x2) for _ in range(x1.size(0))] * 2, 1)
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 2)
