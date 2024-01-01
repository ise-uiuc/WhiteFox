
class Model(torch.nn.Module):
    def __init__(self, min_value=2.5, max_value=0.7):
        super().__init__()
        self.hardtanh = torch.nn.Hardtanh()
        self.conv_transpose = torch.nn.ConvTranspose2d(300, 784, 1, stride=1, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x3):
        v1 = self.conv_transpose(x3)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.hardtanh(v3)
        return v4
# Inputs to the model
x3 = torch.randn(1, 300, 1, 1)
