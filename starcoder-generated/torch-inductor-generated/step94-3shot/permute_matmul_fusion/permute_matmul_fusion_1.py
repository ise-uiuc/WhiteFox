
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        x11 = x1.permute(1, 0, 2)
        x12 = (x1 * x2).permute(1, 0, 2)
        x13 = (x2 * x2).permute(1, 0, 2)
        return (x11, x12, x13)
# Inputs to the model
x1 = torch.randn(2, 2, 1)
x2 = torch.randn(2, 2, 1)
