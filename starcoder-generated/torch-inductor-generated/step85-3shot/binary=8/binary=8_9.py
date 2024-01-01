
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        weight = torch.randn(3, 3, 1, 1, requires_grad=True)
        x3 = torch.nn.functional.conv2d(x1, weight, stride=1, padding=1)
        x4 = x3 + x2
        return x4
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
x2 = torch.randn(2, 3, 64, 64)
