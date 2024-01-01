
class Model(torch.nn.Module):
    def __init__(self, min_value=0.8080811489360046, max_value=-1.4808196063728333):
        super().__init__()
        self.gelu = torch.nn.GELU()
        self.sigmoid = torch.nn.Sigmoid()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=1, bias=True, dilation=3, groups=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.gelu(v3)
        v5 = self.sigmoid(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
