
class Model(torch.nn.Module):
    def __init__(self, min_value=-55.5):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose3d(33, 1, 1, stride=1, padding=0)
        self.min_value = min_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, 99.7)
        return v3
# Inputs to the model
x1 = torch.randn(33, 1, 75, 22, 3)
