
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bilinear_4 = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    def forward(self, x1):
        v1 = self.bilinear_4(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 4, 4)
