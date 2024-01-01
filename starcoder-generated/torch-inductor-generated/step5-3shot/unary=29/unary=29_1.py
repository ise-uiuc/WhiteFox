
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1280, 1280, 1, stride=1)
    def forward(self, x1):
        v1 = torch.clamp_min(self.conv_transpose(x1), -2.1502025604248047)
        v2 = torch.clamp_max(v1, -1.6099610328674316)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1280, 1, 1)
