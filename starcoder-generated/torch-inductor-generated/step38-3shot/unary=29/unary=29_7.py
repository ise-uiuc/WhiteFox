
class Model(torch.nn.Module):
    def __init__(self, min_value=-0.1331, max_value=0.1331):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(2, 4, 5, stride=1, padding=1)
        self.max_pool2d = torch.nn.MaxPool2d(3, stride=1, padding=1)
        self.avg_pool2d = torch.nn.AvgPool2d(3, stride=1, padding=1)
        self.dropout = torch.nn.Dropout2d(-0.7242)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x2):
        v1 = self.conv_transpose(x2)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.max_pool2d(v3)
        v5 = self.avg_pool2d(v3)
        v6 = self.dropout(v5)
        return v6
# Inputs to the model
x2 = torch.randn(1, 2, 32, 32)
