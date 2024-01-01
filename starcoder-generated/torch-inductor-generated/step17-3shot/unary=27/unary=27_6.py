
class Model(torch.nn.Module):
    def __init__(self, min_value=-1.7):
        super().__init__()
        self.conv = torch.nn.Conv1d(9, 2, 1, stride=3)
        self.min_value = min_value
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        return v2
# Inputs to the model
x1 = torch.randn(750, 9)
