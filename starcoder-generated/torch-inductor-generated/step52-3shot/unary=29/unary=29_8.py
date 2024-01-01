
class Model(torch.nn.Module):
    def __init__(self, min_value=0, max_value=0):
        super().__init__()
        self.conv_transpose2d_1 = torch.nn.ConvTranspose2d(6, 3, 4, stride=4, padding=0, bias=False)
        self.conv2d_1 = torch.nn.Conv2d(2, 5, 2, stride=3, padding=2, bias=False)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x3):
        v2 = self.conv_transpose2d_1(x3)
        v1 = self.conv2d_1(v2)
        v3 = torch.clamp_min(v1, self.min_value)
        v4 = torch.clamp_max(v3, self.max_value)
        return v4
# Inputs to the model
x3 = torch.randn(1, 6, 5, 1)
