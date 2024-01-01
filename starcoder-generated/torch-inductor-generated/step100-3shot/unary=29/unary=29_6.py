
class Model(torch.nn.Module):
    def __init__(self, min_value=-9.662e+37, max_value=6.687e+37):
        super().__init__()
        self.conv_transpose2d = torch.nn.ConvTranspose2d(3, 4, 3, stride=2, padding=1)
        self.conv_transpose3d = torch.nn.ConvTranspose3d(2, 5, 3, stride=1, padding=0)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v0 = x1
        v1 = self.conv_transpose2d(v0)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = v3.contiguous(memory_format=torch.channels_last)
        v4 = self.conv_transpose3d(v4)
        return v4
# Inputs to the model
x1 = torch.randn(6, 3, 76, 88, 1)
