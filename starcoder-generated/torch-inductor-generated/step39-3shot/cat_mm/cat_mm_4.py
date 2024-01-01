
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        return torch.stack([x2] * 10, 1)
# Inputs to the model
x1 = torch.randn(10, 2, 3)
x2 = torch.randn(2, 3)
