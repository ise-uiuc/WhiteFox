
class Model(torch.nn.Module):
    def __init__(self, min_value=-0.4, max_value=0.4):
        super().__init__()
        self.relu6 = torch.nn.ReLU6()
        self.conv2d = torch.nn.Conv2d(5, 9, 3, stride=1, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x2):
        v1 = self.conv2d(x2)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.relu6(v3)
        return v4
# Inputs to the model
x2 = torch.randn(1, 5, 56, 56)
