
class Model(torch.nn.Module):
    def __init__(self, min_value=4.6, max_value=1.6):
        super().__init__()
        self.selu = torch.nn.SELU()
        self.max_pool2d = torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 3, 1, stride=1, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x):
        v1 = self.conv_transpose(x)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.selu(v3)
        return v4
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
