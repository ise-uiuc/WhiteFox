
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 2, kernel_size=(11, 11))
    def forward(self, x0):
        v1 = self.conv(x0)
        v2 = v1 - 0.12016621411375035
        return v2
# Inputs to the model
x0 = torch.randn(1, 1, 142, 142)
