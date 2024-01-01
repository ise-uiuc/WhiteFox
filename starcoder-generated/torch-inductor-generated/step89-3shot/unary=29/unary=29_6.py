
class Model(torch.nn.Module):
    def __init__(self, min_value=1.0416, max_value=2.7782):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(5, 13, 1, stride=1)
        self.conv_transpose = torch.nn.ConvTranspose2d(25, 19, 5, stride=3, padding=2)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv_transpose(v1)
        v3 = torch.clamp_min(v2, self.min_value)
        v4 = torch.clamp_max(v3, self.max_value)
        return v4
# Inputs to the model
x1 = torch.randn(1, 5, 3, 3)
