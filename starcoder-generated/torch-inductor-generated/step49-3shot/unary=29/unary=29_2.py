
class Model(torch.nn.Module):
    def __init__(self, min_value=1.7, max_value=5.8):
        super().__init__()
        self.conv_transpose2d = torch.nn.ConvTranspose2d(2, 3, 8, stride=(2, 1), padding=1, output_padding=1, bias=False)
        self.conv2d2 = torch.nn.Conv2d(3, 1, 1, stride=(1, 1), padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x5):
        v1 = self.conv_transpose2d(x5)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.conv2d2(v3)
        return v4
# Inputs to the model
x5 = torch.randn(1, 2, 29, 29)
