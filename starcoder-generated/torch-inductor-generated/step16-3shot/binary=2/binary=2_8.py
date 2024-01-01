
class Model(torch.nn.Module):
    __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, kernel_size=[1, 1], stride=[1, 1])
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 0.1
        return v2
# Inputs to the model
x = torch.randn(1, 1, 2)
