
class Model(torch.nn.Module):
    def __init__(self, min_value=0.1, max_value=0.2):
        super().__init__()
        self.max_pool2d = torch.nn.MaxPool2d(2, stride=1)
        self.dropout = torch.nn.Dropout()
        self.conv_transpose = torch.nn.ConvTranspose2d(12, 8, 3, stride=3, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x4):
        v1 = self.max_pool2d(x4)
        v2 = self.dropout(v1)
        v3 = self.conv_transpose(v2)
        v4 = torch.clamp_min(v3, self.min_value)
        v5 = torch.clamp_max(v4, self.max_value)
        return v5
# Inputs to the model
x4 = torch.randn(1, 12, 11, 21)
