
class Model(torch.nn.Module):
    def __init__(self, min_value=167.4346, max_value=689213254.321):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 3, 3, stride=1, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp_min(v1, self.min_value) # min_value = 167.4346
        v3 = torch.clamp_max(v2, self.max_value) # max_value = 689213254.321
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
