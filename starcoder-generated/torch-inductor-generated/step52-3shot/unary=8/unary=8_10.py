
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(5, 12, 3, stride=3, bias=True, dilation=2, padding=4, output_padding=1)
        self.batch_norm = torch.nn.BatchNorm2d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 + 3
        v3 = torch.clamp(v2, min=0)
        v4 = torch.clamp(v3, max=6)
        v5 = v1 * v4
        v6 = v5 / 6
        return v6
# Inputs to the model
x1 = torch.randn(1, 5, 32, 32)
