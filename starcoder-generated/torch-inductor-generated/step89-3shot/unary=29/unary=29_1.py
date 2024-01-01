
class Model(torch.nn.Module):
    def __init__(self, min_value=-1.0869, max_value=7.5444):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(64, 7, 3, 1, 0)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(7, 3, 5, 2, 0)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = self.conv_transpose2(v1)
        v3 = torch.clamp_min(v2, self.min_value)
        v4 = torch.clamp_max(v3, self.max_value)
        return v4
# Inputs to the model
x1 = torch.randn(4, 64, 5, 12)
