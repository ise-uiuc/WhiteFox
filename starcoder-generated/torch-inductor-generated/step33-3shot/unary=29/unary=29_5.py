
class Model(torch.nn.Module):
    def __init__(self, min_value=0.9832, max_value=-8.6984):
        super().__init__()
        self.avg_pool2d_4 = torch.nn.AvgPool2d(kernel_size=1, stride=1, padding=0)
        self.relu = torch.nn.ReLU(inplace=True)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.avg_pool2d_4(x1)
        v2 = self.relu(v1)
        v3 = torch.clamp_min(v2, self.min_value)
        v4 = torch.clamp_max(v3, self.max_value)
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2)
