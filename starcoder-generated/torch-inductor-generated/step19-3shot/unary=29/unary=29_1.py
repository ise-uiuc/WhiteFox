
class Model(torch.nn.Module):
    def __init__(self, min_value=55.32398057492189, max_value=70.09899604507665):
        super().__init__()
        self.conv_transpose2d = torch.nn.ConvTranspose2d(3, 8, 1, stride=2, bias=True)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose2d(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = v3.permute(0, 3, 2, 1)
        v5 = v4.contiguous()
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
