
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        return torch.clamp(x1, min=0, max=6)
# Inputs to the model
x1 = torch.randn(1, 3, 48, 64)
