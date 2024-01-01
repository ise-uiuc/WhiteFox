
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, kernel_size = 1)
        self.conv2 = torch.nn.Conv2d(3, 3, kernel_size = 1)
    def forward(self, x1):
        v1 = x1.permute(0, 3, 2, 1)
        v3 = x1
        v2 = torch.nn.functional.conv2d(v1, self.conv.weight, self.conv.bias)
        v3 = v3.permute(0, 3, 2, 1)
        v3 = torch.nn.functional.conv2d(v3, self.conv2.weight, self.conv2.bias)
        return v2, v3
# Inputs to the model
x1 = torch.randn(1, 3, 2, 2)
