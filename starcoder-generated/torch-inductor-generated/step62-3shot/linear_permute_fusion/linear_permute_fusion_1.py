
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 2, 2, 1)
    def forward(self, x1):
        v4 = x1
        v1 = torch.nn.functional.conv2d(v4, self.conv.weight, self.conv.bias, 1, (0, 0), (1, 1), (1, 1))
        v2 = v1.permute(0, 3, 1, 2)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2)
