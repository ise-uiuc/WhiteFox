
class Model(torch.nn.Module):
    def __init__(self, min_value=-1.8, max_value=3.7):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 8, 4, stride=4, padding=3)
        self.act_1 = torch.nn.ReLU6()
        self.conv_1 = torch.nn.Conv2d(8, 4, 1, stride=1, padding=0)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv_transpose(v1)
        v3 = torch.clamp_min(v2, self.min_value)
        v4 = torch.clamp_max(v3, self.max_value)
        v5 = self.act_1(v4)
        v7 = self.conv_1(v5)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 112, 112)
