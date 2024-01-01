
class Model(torch.nn.Module):
    def __init__(self, min_value=0.4, max_value=0.5):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=1)
        self.conv_transpose = torch.nn.ConvTranspose2d(16, 5, 1, stride=1, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.softmax(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 16, 16, 16)
