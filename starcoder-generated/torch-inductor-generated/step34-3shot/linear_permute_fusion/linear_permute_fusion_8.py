
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 2, bias=False).requires_grad_(False)
    def forward(self, x1, x2):
        x2 = self.conv(x2)
        return torch.nn.functional.max_pool2d(x1 + x2, 3)
# Inputs to the model
x1 = torch.randn(4, 3, 8, 8, requires_grad=True)
x2 = torch.randn(4, 3, 10, 10)
