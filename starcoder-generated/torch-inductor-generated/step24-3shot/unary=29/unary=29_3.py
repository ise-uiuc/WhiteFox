
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(4, 8, 1, stride=1, padding=1)
    def forward(self, x1, value):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp_min(v1, 5)
        v3 = torch.clamp_max(v2, value)
        return v3
# Inputs to the model
x1 = torch.randn(1, 4, 32, 32)
value = 1
