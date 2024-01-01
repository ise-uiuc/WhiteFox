
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(4, 4, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.selu()
        v3 = F.max_pool2d(v2, 1, 3, padding=3)
        return v3
# Inputs to the model
x1 = torch.randn(1, 4, 64, 64)
