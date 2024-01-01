
class Model(torch.nn.Module):
    def __init__(self, min_value=-22.495158697470076, max_value=28.411253885560395):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(3, 1, 5, stride=5, padding=5)
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 3, stride=2, padding=1)
        self.relu6 = torch.nn.ReLU6()
        self.adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v2 = self.conv_transpose(v1)
        v3 = torch.clamp_min(v2, self.min_value)
        v4 = torch.clamp_max(v3, self.max_value)
        v5 = self.relu6(v4)
        v6 = self.adaptive_avg_pool2d(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
