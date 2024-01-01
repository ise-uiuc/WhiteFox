
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 3, kernel_size=(7, 7))
    def forward(self, x3):
        v1 = self.conv(x3)
        v2 = v1 - -2e-26
        return v2
# Inputs to the model
x3 = torch.randn(1, 8, 66, 66)
