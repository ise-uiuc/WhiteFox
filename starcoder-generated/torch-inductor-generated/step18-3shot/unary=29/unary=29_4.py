
class Model(torch.nn.Module):
    def __init__(self, min_value=0.1, max_value=4.1):
        super().__init__()
        self.hardtanh = torch.nn.Hardtanh()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 1, 1, stride=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x):
        v1 = self.conv_transpose(x)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.hardtanh(v3)
        return v4
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
