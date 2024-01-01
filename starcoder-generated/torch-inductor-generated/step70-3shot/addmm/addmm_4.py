
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3, bias=False)
        self.conv2 = torch.nn.Conv2d(3, 3, 3, bias=False)
    def forward(self, x1, x2):
        v1 = x1.mm(x2)
        v2 = v1 + 1
        v3 = self.conv1(v2)
        v4 = x1 * v3
        v5 = v4.mm(v4)
        v6 = self.conv2(v5)
        return v3
# Inputs to the model
x1 = torch.randn(3, 3, 3, 3, requires_grad=True)
x2 = torch.randn(3, 3, 3, 3, requires_grad=True)
