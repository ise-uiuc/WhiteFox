
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.relu = torch.nn.ReLU(0.2)
        self.dropout = torch.nn.Dropout(0.3, inplace=False)
        self.max_pool2d = torch.nn.MaxPool2d(3, stride=3, padding=0, dilation=1, ceil_mode=False)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x):
        v1 = self.relu(x)
        v2 = self.dropout(v1)
        v3 = self.max_pool2d(v2)
        v4 = torch.clamp_min(v3, self.min_value)
        v5 = torch.clamp_max(v4, self.max_value)
        return v5
min_value = 0.9
max_value = 0.1
# inputs to the model
x = torch.randn(1, 16, 100, 100)
# model ends
