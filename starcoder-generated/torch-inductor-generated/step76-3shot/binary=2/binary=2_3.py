
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, kernel_size=(3, 3))
    def forward(self, x2):
        v1 = self.conv(x2)
        v2 = v1 - 0.1
        return v2
# Inputs to the model
x2 = torch.randn(1, 1, 70, 70)
