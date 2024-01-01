
class Model(torch.nn.Module):
    def __init__(self, min_value=2.954 + 1e-06, max_value=5.057 - 1e-06):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(3, 4, 3, stride=1, padding=1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(4, 8, 3, stride=1, padding=1)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(8, 16, 3, stride=1, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x):
        v1 = self.conv_transpose1(x)
        v2 = self.conv_transpose2(v1)
        v4 = self.conv_transpose3(v2)
        v5 = torch.clamp_min(v4, self.min_value)
        v6 = torch.clamp_max(v5, self.max_value)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 3, 3)
