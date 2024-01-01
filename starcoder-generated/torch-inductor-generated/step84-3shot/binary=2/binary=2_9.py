
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, kernel_size=(20, 15))
    def forward(self, x2):
        v1 = self.conv(x2)
        v2 = v1 - 1.0
        return v2
# Inputs to the model
x2 = torch.randn(1, 1, 58, 35)
