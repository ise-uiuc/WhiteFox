
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.clamp(torch.nn.functional.conv2d(x1, torch.randn(8, 3, 1, 1), torch.randn(8)), min=0, max=6)
        v2 = v1 / 6
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
