
class Model(torch.nn.Module):
    def __init__(self, min_value=0.016, max_value=12.005):
        super().__init__()
        self.convt = torch.nn.ConvTranspose2d(2, 2, 1, stride=1, padding=1)
        self.gelu = torch.nn.GELU()
        self.conv = torch.nn.Conv2d(2, 4, kernel_size=(2, 2), stride=(2, 2), padding=(2, 2))
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.convt(v1)
        v3 = torch.clamp_min(v2, self.min_value)
        v4 = torch.clamp_max(v3, self.max_value)
        v5 = self.gelu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 2, 10, 10)
