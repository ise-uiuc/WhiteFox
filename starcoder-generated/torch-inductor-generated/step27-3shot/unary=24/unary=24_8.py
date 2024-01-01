
class Model(torch.nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.conv = torch.nn.Conv2d(19, 15, kernel_size, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v1 * -1
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.rand(1, 19, 150, 150)
