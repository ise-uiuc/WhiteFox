
class Model(torch.nn.Module):
    def __init__(self, min_value=0.045046454277, max_value=0.0664435716639):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 4, 1, stride=1, padding=0)
        self.conv2d = torch.nn.Conv2d(4, 2, 3, stride=1, padding=2)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp_max(v1, self.max_value)
        v3 = torch.clamp_min(v2, self.min_value)
        v4 = self.conv2d(v3)
        return v4
# Inputs to the model
x1 = torch.randn(2, 3, 69, 69)
