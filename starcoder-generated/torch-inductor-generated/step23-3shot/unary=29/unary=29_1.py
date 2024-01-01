
class Model(torch.nn.Module):
    def __init__(self, min_value=-0.9, max_value=0.9):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 1, stride=2, padding=0)
        self.bn = torch.nn.BatchNorm2d(num_features=8)
        self.tanh = torch.nn.Tanh()
        self.act_1 = torch.nn.ReLU6()
        self.max_pool2d = torch.nn.MaxPool2d(2, stride=1, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.bn(v1)
        v5 = self.act_1(v2)
        v8 = self.max_pool2d(v5)
        v9 = torch.clamp_min(v8, self.min_value)
        v10 = torch.clamp_max(v9, self.max_value)
        v11 = self.tanh(v10)
        return v11
# Inputs to the model
x1 = torch.randn(1, 3, 112, 112)
