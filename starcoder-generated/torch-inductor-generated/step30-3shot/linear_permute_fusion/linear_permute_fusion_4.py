
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)
        self.conv = torch.nn.Conv2d(2, 2, kernel_size=3)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        v3 = torch.nn.functional.conv2d(v2, self.conv.weight, bias=None, stride=1, padding=1, dilation=1, groups=1)
        return v3.permute(0, 2, 3, 1)[:, :, :v2.size(2), :v2.size(3)]
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2)
