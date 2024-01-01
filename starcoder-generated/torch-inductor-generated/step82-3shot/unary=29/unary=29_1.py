
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(17, 9, 7, stride=2, padding=3)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp_min(v1, 0.9096)
        v3 = torch.clamp_max(v2, 0.9089)
        return v3
# Inputs to the model
x1 = torch.randn(1, 17, 93, 52)
