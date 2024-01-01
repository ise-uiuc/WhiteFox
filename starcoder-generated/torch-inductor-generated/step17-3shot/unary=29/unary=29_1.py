
class Model(torch.nn.Module):
    def __init__(self, min_value=-1.8, max_value=3.7):
        super().__init__()
        self.max_pool2d = torch.nn.MaxPool2d(7, stride=1, padding=3)
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=1)
        self.act_1 = torch.nn.ReLU6()
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.act_1(v3)
        v7 = self.max_pool2d(v4)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 112, 112)
