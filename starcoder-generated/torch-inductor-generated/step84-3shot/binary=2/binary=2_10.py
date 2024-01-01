
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, kernel_size=(47, 47))
    def forward(self, x21):
        v1 = self.conv(x21)
        v2 = v1 - 1.0
        return v2
# Inputs to the model
x21 = torch.randn(1, 1, 68, 68)
