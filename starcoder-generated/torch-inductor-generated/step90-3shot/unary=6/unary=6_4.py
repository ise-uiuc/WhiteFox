
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = 3.5 + x1 * 3
        x12 = 6.0 * x2
        x13 = (x12 / 6)
        return x13.unsqueeze(0).unsqueeze(-1)
# Inputs to the model
x1 = torch.randn(3, 3, 64, 64)
