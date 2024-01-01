
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.randn(3, 3)
        v2 = v1 + x1
        c1 = torch.nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        v2 = c1(v2)
        v3 = v1 + x2
        c2 = torch.nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        return c2(v3)
# Inputs to the model
x1 = torch.randn(3, 3, 3, 3, requires_grad=True)
x2 = torch.randn(3, 3, 3, 3, requires_grad=True)
