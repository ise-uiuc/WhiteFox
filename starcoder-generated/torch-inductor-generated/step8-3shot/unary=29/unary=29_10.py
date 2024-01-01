
class Model(torch.nn.Module):
    def __init__(self, min_value=0.1, max_value=3.7):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(3, 1, 3, stride=2, padding=1)
        self.conv1 = torch.nn.Conv2d(1, 2, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(2, 4, 3, stride=2, padding=1)
        self.conv_transpose1 = torch.nn.ConvTranspose2d(4, 2, 3, stride=2, padding=1, output_padding=1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(2, 1, 3, stride=2, padding=1, output_padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x):
        v1 = self.conv0(x)
        v2 = self.conv1(v1)
        v4 = self.conv2(v2)
        v6 = self.conv_transpose1(v4)
        v8 = self.conv_transpose2(v6)
        v9 = torch.clamp_min(v8, self.min_value)
        v10 = torch.clamp_max(v9, self.max_value)
        return v10
# Inputs to the model
x1 = torch.randn(1, 3, 8, 8)
