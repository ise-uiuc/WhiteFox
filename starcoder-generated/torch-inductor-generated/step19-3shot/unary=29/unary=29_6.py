
class Model(torch.nn.Module):
    def __init__(self, min_value=4, max_value=-15):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.max_pool2d = torch.nn.MaxPool2d(kernel_size=[5, 5], stride=1, padding=(0, 1))
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.max_pool2d(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 12, 12)
