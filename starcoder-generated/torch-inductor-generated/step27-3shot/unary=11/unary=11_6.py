
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 16, 3, stride=2, padding=0, dilation=2)
        self.max_pool = torch.nn.MaxPool2d(3, stride=2, padding=0, dilation=2, return_indices=False, ceil_mode=False)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        return self.max_pool(v5)
# Inputs to the model
x1 = torch.randn(1, 3, 96, 96)
