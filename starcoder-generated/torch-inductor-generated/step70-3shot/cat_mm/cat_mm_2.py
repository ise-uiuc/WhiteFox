
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.cat([x, x, x, x], 1)

# Inputs to the model
x = torch.randn(8, 16)
