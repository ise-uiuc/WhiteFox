
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(3, 8, 3, stride=1, padding=2)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(3, 16, 4, stride=1, padding=3)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = self.conv_transpose2(v1)
        v3 = torch.clamp_min(v2, self.min_value)
        v4 = torch.clamp_max(v3, self.max_value)
        return v4
min_value = 0
max_value = 3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
