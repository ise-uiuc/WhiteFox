
class Model(torch.nn.Module):
    def __init__(self, min_value=0.7, max_value=7.3):
        super().__init__()
        self.softmax = torch.nn.Softmax(12)
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 24, 1, stride=1, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x):
        v1 = self.conv_transpose(x)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v5 = self.softmax(v3)
        return v5
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
