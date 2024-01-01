
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.depth_conv = torch.nn.Conv2d(16, 16, 2, stride=(2, 2), padding=1)
    def forward(self, x1):
        v1 = self.depth_conv(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
