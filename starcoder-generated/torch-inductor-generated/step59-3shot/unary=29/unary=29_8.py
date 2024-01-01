
class Model(torch.nn.Module):
    def __init__(self, max_value=0.7):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(15, 33, 5, stride=5, padding=2)
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v2 = torch.clamp_min(v1, -3.16)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3
# Inputs to the model
x1 = torch.randn(1, 15, 99, 101)
