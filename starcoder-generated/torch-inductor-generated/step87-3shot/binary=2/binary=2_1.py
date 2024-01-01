
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 3, kernel_size=(3, 3), stride=1, padding=1, groups=3)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - False
        return v2
# Inputs to the model
x = torch.randn(2, 4, 27, 27)
