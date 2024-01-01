
class Model(torch.nn.Module):
    def __init__(self, min_value=-200, max_value=68):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(5, 10, 2, stride=2, padding=2)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.min_value
        v3 = self.max_value
        v4 = self.conv_transpose(x1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 5, 64, 64)
