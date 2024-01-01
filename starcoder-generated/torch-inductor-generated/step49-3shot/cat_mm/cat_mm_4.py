
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x1 = x1.reshape(-1, 1)
        return torch.cat([x1, x1, x1, x1, x1, x1, x1, x1, x1, x1, x1, x1, x1, x1, x1, x1, x1], 0)
# Inputs to the model
x1 = torch.randn(4, 2)
