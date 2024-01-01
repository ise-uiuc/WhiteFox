
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = torch.nn.Conv2d(33, 64, kernel_size=8, bias=False)
        self.conv2d_2 = torch.nn.Conv2d(64, 64, kernel_size=8, bias=False)
    def forward(self, x1):
        # 1.
        v1 = self.conv2d_1(x1)
        v2 = self.conv2d_2(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 33, 33, 64)
