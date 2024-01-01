
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        with torch.no_grad():
            x1 = torch.nn.functional.conv_transpose2d(x, self.conv_t.weight, None, bias=None, stride=1, dilation=1, groups=1)
        x2 = x1 > 0
        x3 = x1 * 5.398
        x4 = torch.where(x2, x1, x3)
        return x4
# Inputs to the model
x = torch.randn(4, 3, 10, 20)
