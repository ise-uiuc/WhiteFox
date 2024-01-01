
class Model(torch.nn.Module):
    def __init__(self, min_value=-3.2, max_value=0.8):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(15, stride=1, padding=7)
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x0):
        v1 = self.conv_transpose(x0)
        v2 = torch.clamp_max(v1, self.max_value)
        v3 = torch.clamp_min(v2, self.min_value)
        v10 = self.pool(v3)
        return v10
# Inputs to the model
x0 = torch.randn(1, 3, 256, 256)
