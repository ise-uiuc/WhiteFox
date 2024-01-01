
class Model(torch.nn.Module):
    def __init__(self, min_value=-4.321, max_value=3.45):
        super().__init__()
        self.relu6 = torch.nn.ReLU6()
        self.prelu = torch.nn.PReLU()
        self.max_pool2d = torch.nn.MaxPool2d(4, stride=2)
        self.conv_transpose = torch.nn.ConvTranspose2d(2, 4, 3, stride=1, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.max_pool2d(x1)
        v2 = self.conv_transpose(v1)
        v3 = torch.clamp_min(v2, self.min_value)
        v4 = torch.clamp_max(v3, self.max_value)
        v5 = self.prelu(v4)
        v6 = self.relu6(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 2, 4, 4)
