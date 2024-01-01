
class Model(torch.nn.Module):
    def __init__(self, min_value=0, max_value=0):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=1)
        self.conv = torch.nn.Conv2d(8, 3, 1, stride=1, padding=1)
        self.conv_transpose2d = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=1)
        self.relu6 = torch.nn.ReLU6()
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.sigmoid(self.conv_transpose(self.conv_transpose2d(self.relu6(self.conv(x1)))))
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.conv(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
